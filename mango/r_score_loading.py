import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pymongo import MongoClient
from urllib.parse import quote_plus
from ipywidgets import interact, widgets
import warnings
warnings.filterwarnings("ignore")

# -------------------- 1️⃣ Connect to MongoDB --------------------
username = quote_plus("crime_user")
password = quote_plus("ztna")
uri = f"mongodb+srv://{username}:{password}@cluster0.zdvxpf3.mongodb.net/?retryWrites=true&w=majority&appName=cluster0"
client = MongoClient(uri)

# Source DB
db = client["crimeDB"]
collection = db["crime_reports"]

# Analytics DB
analytics_db = client["crimeAnalytics"]
rec_score_collection = analytics_db["recurrence_scores"]

# -------------------- 2️⃣ Load and Clean Data --------------------
cursor = collection.find({})
df = pd.DataFrame(list(cursor))
df = df[["city", "time_of_occurrence", "crime_description"]]
df["crime_description"] = df["crime_description"].astype(str).str.strip()
df = df[df["crime_description"] != ""]

df["date_of_occurrence"] = pd.to_datetime(df["time_of_occurrence"], errors="coerce")
df = df.dropna(subset=["date_of_occurrence"])
df["Month-Year"] = df["date_of_occurrence"].dt.to_period("M")
df["occurred"] = 1

# -------------------- 3️⃣ Create Pivot Table --------------------
pivot = df.pivot_table(
    index=["city", "crime_description"],
    columns="Month-Year",
    values="occurred",
    aggfunc="sum",
    fill_value=0
)

# -------------------- 4️⃣ Define Recurrence Score --------------------
def recurrence_score(row):
    months_with_crime = (row > 0).sum()
    recurrences = ((row.shift(1) > 0) & (row > 0)).sum()
    if months_with_crime <= 1:
        return 0.0
    return recurrences / (months_with_crime - 1)

# -------------------- 5️⃣ Compute and Save Recurrence Scores --------------------
rec_scores = pivot.apply(recurrence_score, axis=1).reset_index()
rec_scores.columns = ["city", "crime_description", "recurrence_score"]
rec_scores = rec_scores.sort_values(by="recurrence_score", ascending=False)

# Save to MongoDB (overwrite)
rec_score_collection.delete_many({})
rec_score_collection.insert_many(rec_scores.to_dict(orient="records"))
print("✅ Recurrence scores stored in MongoDB!")

# -------------------- 6️⃣ Reload from DB --------------------
loaded = list(rec_score_collection.find({}, {"_id": 0}))
df_loaded = pd.DataFrame(loaded)

# -------------------- 7️⃣ Dropdown Filter Visualization --------------------
def plot_city(city):
    topN = 5
    data = df_loaded[df_loaded["city"] == city].nlargest(topN, "recurrence_score")
    
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=data,
        x="recurrence_score",
        y="crime_description",
        palette="crest"
    )
    plt.title(f"Top {topN} Recurring Crimes in {city}", fontsize=16, fontweight="bold")
    plt.xlabel("Recurrence Score (0–1)")
    plt.ylabel("Crime Description")
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

# Interactive dropdown
cities = sorted(df_loaded["city"].unique())
interact(plot_city, city=widgets.Dropdown(options=cities, description="City:"))
