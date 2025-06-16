import pandas as pd
import folium
from pymongo import MongoClient
from urllib.parse import quote_plus
from collections import Counter
from folium import Popup, IFrame

username = quote_plus("crime_user")
password = quote_plus("ztna")
uri = f"mongodb+srv://{username}:{password}@cluster0.zdvxpf3.mongodb.net/?retryWrites=true&w=majority&appName=cluster0"

client = MongoClient(uri)
db = client["crimeDB"]
collection = db["crime_reports"]

# Load relevant data
cursor = collection.find({}, {"city": 1, "crime_description": 1, "location": 1})
df = pd.DataFrame(list(cursor))

def top_crimes(crimes):
    counts = Counter(crimes)
    top5 = [crime for crime, _ in counts.most_common(5)]
    return top5

# Clean data
df["crime_description"] = df["crime_description"].str.strip()
df = df.dropna(subset=["crime_description", "city", "location"])

# Extract coordinates
df["lat"] = df["location"].apply(lambda x: x["coordinates"][1] if isinstance(x, dict) else None)
df["lon"] = df["location"].apply(lambda x: x["coordinates"][0] if isinstance(x, dict) else None)
df = df.dropna(subset=["lat", "lon"])

# Group by city and get top crime pairs (simplified for now: count combos)
city_crime_groups = df.groupby("city")["crime_description"].apply(top_crimes).to_dict()
city_coords = df.groupby("city")[["lat", "lon"]].mean().reset_index()

# Combine and prepare tooltip info
hotspots = []
for _, row in city_coords.iterrows():
    city = row["city"]
    lat, lon = row["lat"], row["lon"]
    crimes = city_crime_groups.get(city, [])
    crime_text = ', '.join(crimes[:5])  # Top 5 for tooltip
    hotspots.append({
        "city": city,
        "lat": lat,
        "lon": lon,
        "tooltip": f"{city}: {crime_text}"
    })

# Plot map
m = folium.Map(location=[22.9734, 78.6569], zoom_start=5)  # Center of India

for _, row in city_coords.iterrows():
    city = row["city"]
    lat, lon = row["lat"], row["lon"]
    crimes = city_crime_groups.get(city, [])

    # Build popup html directly from crimes list
    html = f"""
    <div style="
        font-family: Arial, sans-serif; 
        font-size: 14px; 
        max-width: 250px;
        color: #333;
        background: #fefefe;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        padding: 10px;
    ">
        <h4 style="margin-bottom: 8px; color: #d9534f;">📍 {city}</h4>
        <b>Common Crimes:</b><br>
        <ul style="padding-left: 16px; margin: 0;">
    """

    for crime in crimes:
        html += f"<li>🔸 {crime.title()}</li>"  # Capitalize for presentation

    html += "</ul></div>"

    iframe = IFrame(html=html, width=280, height=150)
    popup = Popup(iframe, max_width=280)

    folium.CircleMarker(
        location=[lat, lon],
        radius=7,
        color="#d9534f",
        fill=True,
        fill_color="#d9534f",
        fill_opacity=0.7,
        popup=popup,
        tooltip=city
    ).add_to(m)

m.save("crime_hotspots.html")
print("✅ Map saved as 'crime_hotspots.html'")
