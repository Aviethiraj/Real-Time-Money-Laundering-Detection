# Class Imbalance Analysis (2.1)
# Bar chart (recommended) + optional pie chart
# -------------------------------------------------
# pip install pandas matplotlib seaborn openpyxl

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1) Load dataset
df = pd.read_excel(r"E:\Download\SAML-D.xlsx", engine="openpyxl")

# 2) Get class counts (0 = Normal, 1 = Laundering)
# Change the column name here if your label column differs
label_col = "Is_laundering"

class_counts = df[label_col].value_counts().sort_index()  # ensures order 0,1
class_names = {0: "Normal", 1: "Laundering"}

# Build a tidy dataframe for plotting
plot_df = pd.DataFrame({
    "Class": [class_names.get(i, str(i)) for i in class_counts.index],
    "Count": class_counts.values
})

# 3) Bar chart (academically preferred)
sns.set(style="whitegrid")
plt.figure(figsize=(8, 5))

ax = sns.barplot(data=plot_df, x="Class", y="Count", palette=["#4C78A8", "#F58518"])
ax.set_title("Class Imbalance: Normal vs Laundering", fontsize=14)
ax.set_xlabel("Class")
ax.set_ylabel("Number of Transactions")

# Add labels on bars
for p in ax.patches:
    ax.annotate(f"{int(p.get_height()):,}",
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha="center", va="bottom", fontsize=11, xytext=(0, 4),
                textcoords="offset points")

plt.tight_layout()
plt.savefig("class_imbalance_bar.png", dpi=300, bbox_inches="tight")
plt.show()

# 4) Optional: Pie chart (less preferred academically, but can be included)
plt.figure(figsize=(6, 6))
plt.pie(
    plot_df["Count"],
    labels=plot_df["Class"],
    autopct=lambda p: f"{p:.4f}%",
    startangle=90,
    colors=["#4C78A8", "#F58518"]
)
plt.title("Class Distribution (Pie Chart)")
plt.tight_layout()
plt.savefig("class_imbalance_pie.png", dpi=300, bbox_inches="tight")
plt.show()

# 5) Print imbalance ratio for reporting
total = plot_df["Count"].sum()
laundering = plot_df.loc[plot_df["Class"] == "Laundering", "Count"].iloc[0]
normal = plot_df.loc[plot_df["Class"] == "Normal", "Count"].iloc[0]

print(f"Total transactions: {total:,}")
print(f"Normal: {normal:,} ({normal/total:.6%})")
print(f"Laundering: {laundering:,} ({laundering/total:.6%})")
print(f"Imbalance ratio (Normal:Laundering) = {normal/laundering:.2f}:1")