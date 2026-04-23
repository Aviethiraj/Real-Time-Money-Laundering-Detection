# ============================================================
# 3.2.3 Account and Network Statistics
# - Degree distribution plot (log-log) to show power-law behavior
# - Optional: small subgraph network diagram (e.g., mule ring)
# ============================================================

# Install (if needed):
# pip install pandas numpy matplotlib networkx openpyxl powerlaw

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Optional power-law fitting (recommended for dissertation-quality evidence)
try:
    import powerlaw
    POWERLAW_AVAILABLE = True
except ImportError:
    POWERLAW_AVAILABLE = False
    print("Optional package missing: powerlaw. Install with: pip install powerlaw")

# -----------------------
# 1) Load your dataset
# -----------------------
FILE_PATH = r"E:\Download\SAML-D.xlsx"   # change if needed

df = pd.read_excel(FILE_PATH, engine="openpyxl")

# Ensure columns exist (edit names if your file differs)
SENDER_COL = "Sender_account"
RECEIVER_COL = "Receiver_account"

assert SENDER_COL in df.columns and RECEIVER_COL in df.columns, "Sender/Receiver columns not found."

# Optional: if your accounts are numeric, cast to string to avoid issues in NetworkX labels
df[SENDER_COL] = df[SENDER_COL].astype(str)
df[RECEIVER_COL] = df[RECEIVER_COL].astype(str)

# -----------------------
# 2) Build transaction graph
# -----------------------
# Directed graph is typical for money flow (sender -> receiver)
G = nx.from_pandas_edgelist(
    df,
    source=SENDER_COL,
    target=RECEIVER_COL,
    create_using=nx.DiGraph()
)

print(f"Graph built: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

# -----------------------
# 3) Degree distribution (log-log)
# -----------------------
# For directed graphs you can analyze:
# - total degree (in+out)
# - in-degree
# - out-degree
degrees_total = np.array([d for _, d in G.degree()])
degrees_in = np.array([d for _, d in G.in_degree()])
degrees_out = np.array([d for _, d in G.out_degree()])

def plot_degree_distribution_loglog(deg_array, title, out_path=None):
    deg_array = deg_array[deg_array > 0]  # remove zeros for log scale

    # Count frequency for each degree value
    unique_deg, counts = np.unique(deg_array, return_counts=True)

    plt.figure(figsize=(7, 5))
    plt.scatter(unique_deg, counts, s=12, alpha=0.75)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Degree (k) [log scale]")
    plt.ylabel("Frequency P(k) [log scale]")
    plt.title(title)
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {out_path}")
    plt.show()

# Plot: total / in / out (choose what you want to include in dissertation)
plot_degree_distribution_loglog(degrees_total, "Degree Distribution (Total Degree) - Log-Log",
                                out_path="degree_distribution_total_loglog.png")
plot_degree_distribution_loglog(degrees_in, "Degree Distribution (In-Degree) - Log-Log",
                                out_path="degree_distribution_in_loglog.png")
plot_degree_distribution_loglog(degrees_out, "Degree Distribution (Out-Degree) - Log-Log",
                                out_path="degree_distribution_out_loglog.png")

# -----------------------
# 4) Optional: power-law fit (adds strong academic justification)
# -----------------------
if POWERLAW_AVAILABLE:
    # powerlaw expects 1D array of observations; remove zeros
    data = degrees_total[degrees_total > 0]

    fit = powerlaw.Fit(data, discrete=True, verbose=False)
    alpha = fit.power_law.alpha
    xmin = fit.power_law.xmin

    print("\nPower-law fit (Total Degree)")
    print(f"alpha (scaling exponent): {alpha:.3f}")
    print(f"xmin: {xmin}")

    # Compare power-law vs lognormal (common in real networks)
    R, p = fit.distribution_compare("power_law", "lognormal")
    print(f"Compare power_law vs lognormal: R={R:.3f}, p={p:.4f}")
    print("Interpretation: if p < 0.05, comparison is statistically meaningful; sign of R indicates preferred model.")

    # Plot CCDF for clean power-law style figure
    plt.figure(figsize=(7, 5))
    fit.plot_ccdf(label="Empirical CCDF", color="blue")
    fit.power_law.plot_ccdf(label="Fitted power-law", color="red", linestyle="--")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Degree k [log scale]")
    plt.ylabel("P(K ≥ k) [log scale]")
    plt.title("Degree CCDF with Power-law Fit (Total Degree)")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig("degree_ccdf_powerlaw_fit.png", dpi=300, bbox_inches="tight")
    print("Saved: degree_ccdf_powerlaw_fit.png")
    plt.show()

# -----------------------
# 5) Optional network diagram: small subgraph example (mule ring illustration)
# -----------------------
# Approach:
#  - find a dense-ish local neighborhood around a “high out-degree” node
#  - then draw a small induced subgraph
#
# You can also restrict to suspicious-only edges if you have labels (e.g., Is_laundering == 1).
# For now, we show a general illustration.

# Pick a seed node with high out-degree (potential hub)
top_out = sorted(G.out_degree(), key=lambda x: x[1], reverse=True)
seed_node = top_out[0][0]
print(f"\nSeed node (highest out-degree): {seed_node}, out-degree={top_out[0][1]}")

# Take its 1-hop neighbors (outgoing) and optionally 2-hop to make it more illustrative
neighbors_1 = set(G.successors(seed_node))
nodes_sub = set([seed_node]) | neighbors_1

# Cap size so the plot stays readable
MAX_NODES = 60
if len(nodes_sub) > MAX_NODES:
    nodes_sub = set([seed_node]) | set(list(neighbors_1)[:MAX_NODES-1])

H = G.subgraph(nodes_sub).copy()
print(f"Subgraph: {H.number_of_nodes()} nodes, {H.number_of_edges()} edges")

plt.figure(figsize=(9, 7))

# Layout options: spring_layout is fine for small subgraphs
pos = nx.spring_layout(H, seed=42, k=0.6)

# Node styling: highlight seed node
node_colors = ["red" if n == seed_node else "skyblue" for n in H.nodes()]
node_sizes = [600 if n == seed_node else 200 for n in H.nodes()]

nx.draw_networkx_nodes(H, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9)
nx.draw_networkx_edges(H, pos, arrowstyle="->", arrowsize=12, width=1.0, alpha=0.5)

# Labels can get messy; enable only if subgraph is small
if H.number_of_nodes() <= 30:
    nx.draw_networkx_labels(H, pos, font_size=7)

plt.title("Example Transaction Subgraph (Hub-and-Spoke Illustration)")
plt.axis("off")
plt.tight_layout()
plt.savefig("example_subgraph_network.png", dpi=300, bbox_inches="tight")
print("Saved: example_subgraph_network.png")
plt.show()

