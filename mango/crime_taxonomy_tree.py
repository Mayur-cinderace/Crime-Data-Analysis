import pandas as pd
import numpy as np
from itertools import combinations
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from pymongo import MongoClient
from urllib.parse import quote_plus
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import seaborn as sns
import networkx as nx
import plotly.graph_objects as go

username = quote_plus("crime_user")
password = quote_plus("ztna")
uri = f"mongodb+srv://{username}:{password}@cluster0.zdvxpf3.mongodb.net/?retryWrites=true&w=majority&appName=cluster0"
client = MongoClient(uri)
db = client["crimeDB"]
collection = db["crime_reports"]

# -------------------- 📥 Load & Clean Data --------------------
cursor = collection.find({})
df = pd.DataFrame(list(cursor))
df = df[["city", "time_of_occurrence", "crime_description"]]
df["crime_description"] = df["crime_description"].astype(str).str.strip()
df = df[df["crime_description"] != ""]
df["date_of_occurrence"] = pd.to_datetime(df["time_of_occurrence"], errors='coerce')
df = df.dropna(subset=["date_of_occurrence"])
df["Month-Year"] = df["date_of_occurrence"].dt.to_period("M")

# -------------------- 🌲 Crime Taxonomy Dendrogram --------------------
crime_types = df["crime_description"].unique()
if len(crime_types) == 0:
    raise ValueError("❌ No crime types found. Check your data.")

crime_index = {crime: i for i, crime in enumerate(crime_types)}
n = len(crime_types)
co_matrix = np.zeros((n, n))

grouped = df.groupby(["city", "Month-Year"])
for _, group in grouped:
    crimes_in_group = group["crime_description"].unique()
    for c1, c2 in combinations(crimes_in_group, 2):
        i, j = crime_index[c1], crime_index[c2]
        co_matrix[i][j] += 1
        co_matrix[j][i] += 1
    for c in crimes_in_group:
        i = crime_index[c]
        co_matrix[i][i] += 1

if co_matrix.max() == 0:
    raise ValueError("❌ Co-occurrence matrix is all zeros. Grouping might be too fine or data is sparse.")

# Normalize co-occurrence to distances
co_matrix_norm = co_matrix / co_matrix.max()
distance_matrix = 1 - co_matrix_norm

# Hierarchical clustering
linked = linkage(distance_matrix, method='ward')
# Set modern minimalist theme
plt.style.use("seaborn-v0_8-white")

# Color palette for consistency
color_palette = sns.color_palette("coolwarm", 10)

# Plot
fig, ax = plt.subplots(figsize=(16, 10), facecolor='white')

dendrogram(
    linked,
    labels=crime_types,
    orientation='right',
    leaf_font_size=11,
    leaf_rotation=0,
    color_threshold=0.7,
    above_threshold_color=color_palette[3],
    ax=ax
)

# Modern clean title and labels
ax.set_title("Modern Crime Taxonomy Dendrogram", fontsize=20, fontweight='bold', color='#222')
ax.set_xlabel("Dissimilarity", fontsize=14, labelpad=15)
ax.set_ylabel("Crime Types", fontsize=14, labelpad=15)

# Remove borders/spines for modern look
sns.despine(offset=10, trim=True)

# Clean grid
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)

# Fonts and ticks
plt.xticks(fontsize=12)
plt.yticks(fontsize=11)

# Tight layout
plt.tight_layout()
plt.show()

# -------------------- 🧮 Apriori Rule Mining --------------------
transactions = []
for _, group in grouped:
    crimes = list(group["crime_description"].dropna().unique())
    if len(crimes) > 1:
        transactions.append(crimes)

print(f"🧾 Prepared {len(transactions)} transactions.")

te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_array, columns=te.columns_)

# Run Apriori
frequent_itemsets = apriori(df_encoded, min_support=0.05, use_colnames=True)

# Generate Rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
rules = rules.sort_values("lift", ascending=False)

# Step 1: Build the graph
G = nx.DiGraph()
max_rules = 10
rules = rules.sort_values("lift", ascending=False).head(max_rules)

for _, row in rules.iterrows():
    antecedents = list(row['antecedents'])
    consequents = list(row['consequents'])

    ant_str = ', '.join(antecedents)
    con_str = ', '.join(consequents)
    sentence = (
        f"⚠️ When crimes like {ant_str} are reported in an area, "
        f"there's a noticeable pattern of {con_str} happening soon after."
    )
    for a in antecedents:
        for c in consequents:
            G.add_edge(a, c, text=sentence, weight=row['lift'])

# Step 2: Get node positions using spring layout
pos = nx.spring_layout(G, k=0.8, iterations=100)

# Step 3: Build Plotly edge traces with hover
edge_x = []
edge_y = []
edge_hover_trace_x = []
edge_hover_trace_y = []
edge_text = []

for edge in G.edges(data=True):
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]

    # Midpoint for hover
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    edge_hover_trace_x.append(mx)
    edge_hover_trace_y.append(my)
    edge_text.append(edge[2]['text'])

# Line trace (edges)
edge_trace = go.Scatter(
    x=edge_x,
    y=edge_y,
    line=dict(width=1, color='gray'),
    hoverinfo='none',
    mode='lines',
    name='Edges'
)

# Invisible points at edge midpoints for tooltips
edge_hover_trace = go.Scatter(
    x=edge_hover_trace_x,
    y=edge_hover_trace_y,
    mode='markers',
    marker=dict(
        size=8,
        color='lightcoral',
        opacity=0.3,
        line=dict(width=1, color='red')
    ),
    hoverinfo='text',
    text=edge_text,
    name='Crime Rule Insights'
)

# Step 4: Node trace
node_x = []
node_y = []
node_text = []
for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)
    node_text.append(node)

node_trace = go.Scatter(
    x=node_x,
    y=node_y,
    mode='markers+text',
    text=node_text,
    textposition='top center',
    hoverinfo='text',
    marker=dict(
        showscale=False,
        color='lightblue',
        size=20,
        line=dict(width=2, color='black')
    )
)

# Step 5: Combine into figure
fig = go.Figure(data=[edge_trace, edge_hover_trace, node_trace],
                layout=go.Layout(
                    title='🧠 Crime Association Rules Graph',
                    titlefont_size=20,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=10, r=10, t=40),
                    xaxis=dict(showgrid=False, zeroline=False),
                    yaxis=dict(showgrid=False, zeroline=False),
                ))

fig.show()
