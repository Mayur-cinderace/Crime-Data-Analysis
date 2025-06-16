from pymongo import MongoClient
import pandas as pd
from urllib.parse import quote_plus
from collections import defaultdict
import random
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# ----- MongoDB connection and data loading -----
username = quote_plus("crime_user")
password = quote_plus("ztna")  # ensure correct credentials
uri = (
    f"mongodb+srv://{username}:{password}@cluster0.zdvxpf3.mongodb.net/"
    "?retryWrites=true&w=majority&appName=cluster0"
)
client = MongoClient(uri)
db = client["crimeDB"]
collection = db["crime_reports"]

# ----- Load and clean data -----
df = pd.DataFrame(list(collection.find({}, {
    "victim_age": 1,
    "crime_description": 1,
    "time_of_occurrence": 1,
    "_id": 0
})))
df["Date"] = pd.to_datetime(df["time_of_occurrence"], errors='coerce')
df = df.dropna(subset=["victim_age", "Date"]).sort_values("Date")

# ----- Age grouping -----
bins = [0, 12, 18, 25, 35, 50, 65, 100]
labels = ["Child", "Teen", "Youth", "Adult", "MidAge", "Senior", "Elder"]
df["Age Group"] = pd.cut(df["victim_age"], bins=bins, labels=labels)

# ----- Transition probability builder -----
def build_transition_probs(subdf):
    crimes = subdf.sort_values("Date")["crime_description"].tolist()
    pairs = [
        (crimes[i], crimes[i+1])
        for i in range(len(crimes)-1)
        if crimes[i] != crimes[i+1]
    ]
    matrix = defaultdict(lambda: defaultdict(int))
    for frm, to in pairs:
        matrix[frm][to] += 1
    return {frm: {t: cnt/sum(tos.values()) for t, cnt in tos.items()} for frm, tos in matrix.items()}

# ----- Build probabilities -----
all_probs = build_transition_probs(df)
per_group_probs = {grp: build_transition_probs(df[df["Age Group"] == grp]) for grp in labels}

# ----- Abbreviations -----
all_crimes = set()
for probs in list(per_group_probs.values()) + [all_probs]:
    for frm, tos in probs.items():
        all_crimes.add(frm)
        all_crimes.update(tos.keys())
abbr_map = {}
used = set()
for crime in sorted(all_crimes):
    for ch in crime.upper():
        if ch.isalpha() and ch not in used:
            abbr_map[crime] = ch
            used.add(ch)
            break
    else:
        for d in '0123456789':
            if d not in used:
                abbr_map[crime] = d
                used.add(d)
                break

# ----- Prepare top transitions -----
top_n = 3
rows = []
for grp, probs in per_group_probs.items():
    for frm, tos in probs.items():
        for to, p in tos.items():
            rows.append({
                "Group": grp,
                "Transition": f"{abbr_map[frm]}→{abbr_map[to]}",
                "Prob": p
            })
combined_df = pd.DataFrame(rows)
top_combined = combined_df.groupby('Group').apply(
    lambda d: d.nlargest(top_n, 'Prob')
).reset_index(drop=True)

# ----- Animated grouped bar chart -----
transitions = top_combined['Transition'].unique().tolist()
buckets = labels
x = np.arange(len(transitions))
width = 0.8 / len(buckets)

fig, ax = plt.subplots(figsize=(12, 6))
bars = {}
for idx, grp in enumerate(buckets):
    bars[grp] = ax.bar(
        x + (idx - (len(buckets)-1)/2)*width,
        np.zeros(len(transitions)), width,
        label=grp, alpha=0.8, edgecolor='white'
    )
ax.set_xticks(x)
ax.set_xticklabels(transitions, rotation=0, ha='center')
ax.set_ylabel('Probability', labelpad=10)
ax.set_title(f'Top {top_n} Transitions by Age Group', pad=20)
ax.legend(title='Age Group', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.set_ylim(0, top_combined['Prob'].max()*1.2)
plt.tight_layout()

# Animation function

def animate(frame):
    fraction = frame / 30
    for grp in buckets:
        subset = top_combined[top_combined['Group']==grp].set_index('Transition')
        target = [subset['Prob'].get(t,0) for t in transitions]
        for bar, height in zip(bars[grp], target):
            bar.set_height(height * fraction)
    return [b for bars_grp in bars.values() for b in bars_grp]

ani = animation.FuncAnimation(
    fig, animate, frames=30, interval=50, blit=True, repeat=False
)

# ----- Redesigned plot_transition_graph -----
def plot_transition_graph(probs, title="Crime Transition Graph", top_n=10):
    # Build directed graph
    G = nx.DiGraph()
    edges = sorted(
        ((f, t, p) for f, ts in probs.items() for t, p in ts.items()),
        key=lambda x: x[2], reverse=True
    )[:top_n]
    for f, t, p in edges:
        G.add_edge(f, t, weight=p)

    # Layout and styling
    pos = nx.spring_layout(G, k=0.4, seed=42)
    weights = [G[u][v]['weight']*10 for u, v in G.edges()]
    node_sizes = [800 + 200*G.degree(n) for n in G.nodes()]

    plt.figure(figsize=(10, 7))
    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color='skyblue',
        alpha=0.9,
        linewidths=1,
        edgecolors='black'
    )
    nx.draw_networkx_edges(
        G, pos,
        width=weights,
        arrowstyle='-|>',
        arrowsize=20,
        alpha=0.7,
        edge_color='gray'
    )
    nx.draw_networkx_labels(
        G, pos,
        font_size=10,
        font_color='black'
    )

    # Annotate edge weights
    for (u, v, d) in G.edges(data=True):
        x_mid = (pos[u][0] + pos[v][0]) / 2
        y_mid = (pos[u][1] + pos[v][1]) / 2
        plt.text(
            x_mid, y_mid,
            f"{d['weight']:.2f}",
            fontsize=8,
            ha='center',
            va='center',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
        )

    plt.title(title, fontsize=14, pad=15)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# ----- Top 5 Crimes Frequency (static) -----
bucket = "Youth"
freq = df[df['Age Group']==bucket]['crime_description'].value_counts().nlargest(5)
labels_freq = freq.index.tolist()
values_freq = freq.values.tolist()

# Create figure and bars
fig2, ax2 = plt.subplots(figsize=(8, 5))
bars2 = ax2.bar(labels_freq, [0]*len(values_freq), color='skyblue', edgecolor='white')
ax2.set_title(f'Top 5 Crimes — {bucket}', fontsize=14, pad=15)
ax2.set_ylabel('Count', labelpad=10)
ax2.set_ylim(0, max(values_freq)*1.2)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Animation function for freq

def animate_freq(frame):
    fraction = frame / 30
    for bar, target in zip(bars2, values_freq):
        bar.set_height(target * fraction)
    return bars2

ani2 = animation.FuncAnimation(fig2, animate_freq, frames=30, interval=50, blit=True, repeat=False)

plt.show()

random.seed(42)

def simulate_path(probs, start, steps=5):
    path=[start]
    for _ in range(steps):
        tos=probs.get(path[-1],{})
        if not tos: break
        path.append(random.choices(list(tos), weights=tos.values())[0])
    return path

def most_probable_path(probs, start, steps=5):
    path=[start]
    for _ in range(steps):
        tos=probs.get(path[-1],{})
        if not tos: break
        path.append(max(tos, key=tos.get))
    return path

# ----- Redesigned network graph -----
def plot_transition_graph(probs, title="Crime Transition Network", top_n=8):
    """
    Displays a clean, modern network graph with large nodes and clear arrowheads.
    """
    G = nx.DiGraph()
    edges = sorted(
        ((f, t, p) for f, ts in probs.items() for t, p in ts.items()),
        key=lambda x: x[2], reverse=True
    )[:top_n]
    for f, t, p in edges:
        G.add_edge(f, t, weight=p)

    pos = nx.spring_layout(G, seed=42)  # modern organic layout
    plt.figure(figsize=(11, 8), facecolor='white')

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_size=2800,
        node_color='#FFE873',
        edgecolors='#1f78b4',
        linewidths=2,
        alpha=0.95
    )

    # Draw edges with arrowheads
    nx.draw_networkx_edges(
        G, pos,
        edge_color='#4B8BBE',
        width=[G[u][v]['weight'] * 5 for u, v in G.edges()],
        arrows=True,
        arrowstyle='-|>',
        arrowsize=25,
        alpha=0.8,
        connectionstyle='arc3,rad=0.15'
    )

    # Draw labels (abbreviated)
    labels = {n: abbr_map.get(n, n[:1].upper()) for n in G.nodes()}
    nx.draw_networkx_labels(
        G, pos,
        labels=labels,
        font_size=16,
        font_color='black',
        font_weight='bold'
    )

    plt.title(title, fontsize=20, pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# Example usage
start="VANDALISM"
print("Random path:", simulate_path(all_probs, start))
print("Most probable path:", most_probable_path(all_probs, start))
plot_transition_graph(all_probs)
