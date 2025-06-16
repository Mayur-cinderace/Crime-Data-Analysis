import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pymongo import MongoClient
from urllib.parse import quote_plus

# -------------------- 1️⃣ Connect to MongoDB --------------------
username = quote_plus("crime_user")
password = quote_plus("ztna")
uri = f"mongodb+srv://{username}:{password}@cluster0.zdvxpf3.mongodb.net/?retryWrites=true&w=majority&appName=cluster0"
client = MongoClient(uri)

# Connect to stored recurrence score DB
rec_score_collection = client["crimeAnalytics"]["recurrence_scores"]

# -------------------- 2️⃣ Load Recurrence Scores --------------------
loaded = list(rec_score_collection.find({}, {"_id": 0}))
df_loaded = pd.DataFrame(loaded)

if df_loaded.empty:
    print("⚠️ No recurrence data found in MongoDB.")
    exit()

print(f"✅ Loaded {len(df_loaded)} recurrence records from MongoDB.")

# -------------------- 3️⃣ City Selection via CLI --------------------
cities = sorted(df_loaded["city"].unique())

print("\nAvailable Cities:")
for idx, city in enumerate(cities):
    print(f"{idx + 1}. {city}")

choice = input("\nEnter the number of the city you want to view: ")
try:
    choice_idx = int(choice) - 1
    if 0 <= choice_idx < len(cities):
        selected_city = cities[choice_idx]
    else:
        print("❌ Invalid selection.")
        exit()
except ValueError:
    print("❌ Please enter a valid number.")
    exit()

# -------------------- 4️⃣ Plot for Selected City --------------------
topN = 5
data = df_loaded[df_loaded["city"] == selected_city].nlargest(topN, "recurrence_score")

plt.figure(figsize=(10, 6))
sns.barplot(
    data=data,
    x="recurrence_score",
    y="crime_description",
    palette="crest"
)
plt.title(f"Top {topN} Recurring Crimes in {selected_city}", fontsize=16, fontweight="bold")
plt.xlabel("Recurrence Score (0–1)")
plt.ylabel("Crime Description")
plt.grid(axis="x", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()
