import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score,
                              confusion_matrix, classification_report)

print("=" * 65)
print("  E-COMMERCE CUSTOMER BEHAVIOR ANALYSIS")
print("=" * 65)
print("\n[STEP 1] Generating Synthetic Dataset...")

np.random.seed(42)
N = 500

dates = pd.date_range(start="2023-01-01", end="2023-12-31", periods=N)

customer_ids = np.random.randint(1001, 1201, size=N)

categories    = ["Electronics", "Clothing", "Books", "Home & Kitchen", "Sports"]
product_ids   = np.random.choice(range(1, 21), size=N)
product_cats  = np.random.choice(categories, size=N)

quantity   = np.random.randint(1, 10, size=N)
unit_price = np.round(np.random.uniform(5.0, 500.0, size=N), 2)
revenue    = np.round(quantity * unit_price, 2)

age              = np.random.randint(18, 65, size=N)
total_spent      = np.round(np.random.uniform(50, 5000, size=N), 2)
num_prev_orders  = np.random.randint(0, 20, size=N)
avg_rating       = np.round(np.random.uniform(1.0, 5.0, size=N), 1)
days_since_last  = np.random.randint(1, 365, size=N)

repeat_buyer = (num_prev_orders > 5).astype(int)

df = pd.DataFrame({
    "Date"           : dates,
    "Customer_ID"    : customer_ids,
    "Product_ID"     : product_ids,
    "Category"       : product_cats,
    "Quantity"       : quantity,
    "Unit_Price"     : unit_price,
    "Revenue"        : revenue,
    "Age"            : age,
    "Total_Spent"    : total_spent,
    "Num_Prev_Orders": num_prev_orders,
    "Avg_Rating"     : avg_rating,
    "Days_Since_Last": days_since_last,
    "Repeat_Buyer"   : repeat_buyer,
})

df.to_csv("ecommerce_data.csv", index=False)
print("  Dataset saved as 'ecommerce_data.csv'")

print("\n[STEP 2] Dataset Overview")
print("-" * 40)
print(df.head(5).to_string())
print(f"\nShape          : {df.shape}")
print(f"Columns        : {list(df.columns)}")
print(f"\nMissing Values :\n{df.isnull().sum()}")

print("\n[STEP 3] Descriptive Statistics")
print("-" * 40)
numeric_cols = ["Quantity", "Unit_Price", "Revenue",
                "Age", "Total_Spent", "Num_Prev_Orders",
                "Avg_Rating", "Days_Since_Last"]
desc = df[numeric_cols].describe().round(2)
print(desc.to_string())

total_revenue    = df["Revenue"].sum()
avg_order_value  = df["Revenue"].mean()
total_customers  = df["Customer_ID"].nunique()
repeat_rate      = df["Repeat_Buyer"].mean() * 100

print(f"\n  Total Revenue Generated     : ₹{total_revenue:,.2f}")
print(f"  Average Order Value (AOV)   : ₹{avg_order_value:,.2f}")
print(f"  Unique Customers            : {total_customers}")
print(f"  Repeat Buyer Rate           : {repeat_rate:.1f}%")

print("\n[STEP 4] Computing & Plotting Correlation Matrix...")

corr_cols = ["Quantity", "Unit_Price", "Revenue",
             "Age", "Total_Spent", "Num_Prev_Orders",
             "Avg_Rating", "Days_Since_Last"]
corr_matrix = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5,
    ax=ax,
    cbar=True,
)
ax.set_title("Correlation Matrix – E-Commerce Features", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("plot_1_correlation_heatmap.png", dpi=120)
plt.close()
print("  Saved → plot_1_correlation_heatmap.png")

print("\n  Correlation Matrix:")
print(corr_matrix.round(3).to_string())

print("""
  Interpretation:
  1. Diagonal values = 1.00 (perfect self-correlation).
  2. Revenue is strongly correlated with Unit_Price (higher price → higher revenue).
  3. Num_Prev_Orders weakly correlates with Total_Spent (loyal customers spend more).
  4. Age and Quantity show near-zero correlation (age doesn't drive quantity).
""")

print("[STEP 5] Analyzing Monthly Revenue Trend...")

df["Month"] = df["Date"].dt.to_period("M")
monthly_revenue = df.groupby("Month")["Revenue"].sum()

fig, ax = plt.subplots(figsize=(12, 5))
monthly_revenue.plot(kind="line", marker="o", color="steelblue", linewidth=2, ax=ax)
ax.set_title("Monthly Revenue Trend (Jan–Dec 2023)", fontsize=13, fontweight="bold")
ax.set_xlabel("Month")
ax.set_ylabel("Total Revenue (₹)")
ax.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("plot_2_monthly_revenue.png", dpi=120)
plt.close()
print("  Saved → plot_2_monthly_revenue.png")

print("\n  Monthly Revenue Summary:")
for period, rev in monthly_revenue.items():
    print(f"    {period} : ₹{rev:>10,.2f}")

print("\n[STEP 6] Product & Category Revenue Analysis...")

product_revenue = df.groupby("Product_ID")["Revenue"].sum().sort_values(ascending=False)
top_product     = product_revenue.idxmax()
print(f"  Product with Highest Revenue  : Product_ID {top_product}  "
      f"(₹{product_revenue.max():,.2f})")

avg_qty = df["Quantity"].mean()
print(f"  Average Quantity per Transaction : {avg_qty:.2f} units")

cat_revenue = df.groupby("Category")["Revenue"].sum().sort_values(ascending=False)
print("\n  Category-wise Revenue:")
for cat, rev in cat_revenue.items():
    print(f"    {cat:<20} : ₹{rev:>10,.2f}")

fig, ax = plt.subplots(figsize=(9, 5))
cat_revenue.plot(kind="bar", color="coral", edgecolor="black", ax=ax)
ax.set_title("Revenue by Product Category", fontsize=13, fontweight="bold")
ax.set_xlabel("Category")
ax.set_ylabel("Total Revenue (₹)")
ax.tick_params(axis='x', rotation=30)
plt.tight_layout()
plt.savefig("plot_3_category_revenue.png", dpi=120)
plt.close()
print("  Saved → plot_3_category_revenue.png")

fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(data=df, x="Unit_Price", y="Quantity",
                hue="Category", alpha=0.6, ax=ax)
ax.set_title("Correlation: Quantity Sold vs Unit Price", fontsize=13, fontweight="bold")
ax.set_xlabel("Unit Price (₹)")
ax.set_ylabel("Quantity Sold")
ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("plot_4_qty_vs_price.png", dpi=120)
plt.close()
corr_val = df[["Quantity", "Unit_Price"]].corr().iloc[0, 1]
print(f"\n  Correlation (Quantity vs Unit Price) : {corr_val:.4f}")
print("  Saved → plot_4_qty_vs_price.png")

print("\n[STEP 7] Z-Test: Average Order Value Hypothesis Test")
print("-" * 50)
print("  H₀ : Mean Average Order Value = ₹1200")
print("  H₁ : Mean Average Order Value > ₹1200  (one-tailed)")
print("  α  : 0.05\n")

sample          = df["Revenue"].values
sample_mean     = np.mean(sample)
sample_std      = np.std(sample, ddof=1)
sample_size     = len(sample)
null_mean       = 1200.0
sig_level       = 0.05

standard_error  = sample_std / np.sqrt(sample_size)
z_score         = (sample_mean - null_mean) / standard_error
critical_z      = norm.ppf(1 - sig_level)
p_value         = 1 - norm.cdf(z_score)

decision = "Reject H₀" if z_score > critical_z else "Fail to Reject H₀"

print(f"  Sample Size           : {sample_size}")
print(f"  Sample Mean (AOV)     : ₹{sample_mean:,.2f}")
print(f"  Sample Std Dev        : ₹{sample_std:,.2f}")
print(f"  Standard Error        : ₹{standard_error:,.2f}")
print(f"  Z-Score               : {z_score:.4f}")
print(f"  Critical Z (α=0.05)   : {critical_z:.4f}")
print(f"  P-Value               : {p_value:.4f}")
print(f"\n  Decision              : {decision}")
if z_score > critical_z:
    print("  → The AOV is significantly greater than ₹1200.")
else:
    print("  → No sufficient evidence that AOV exceeds ₹1200.")

print("\n[STEP 8] NumPy Array Operations")
print("-" * 50)

rev_array = df["Revenue"].values
print(f"  Revenue Array Shape : {rev_array.shape}")
print(f"  Mean Revenue        : ₹{np.mean(rev_array):,.2f}")
print(f"  Sum  Revenue        : ₹{np.sum(rev_array):,.2f}")
print(f"  Min  Revenue        : ₹{np.min(rev_array):,.2f}")
print(f"  Max  Revenue        : ₹{np.max(rev_array):,.2f}")
print(f"  Std  Revenue        : ₹{np.std(rev_array):,.2f}")

reshaped = rev_array[:500].reshape(50, 10)
print(f"\n  Reshaped Array      : {reshaped.shape}  (50 rows × 10 cols)")

transposed = reshaped.T
print(f"  Transposed Array    : {transposed.shape}  (10 rows × 50 cols)")

print("\n  NaN Handling Demo:")
nan_array = reshaped.astype(float).copy()
rand_rows = np.random.randint(0, 50, 20)
rand_cols = np.random.randint(0, 10, 20)
for r, c in zip(rand_rows, rand_cols):
    nan_array[r, c] = np.nan

nan_count_before = np.sum(np.isnan(nan_array))
print(f"    NaN count (before replacement) : {nan_count_before}")

col_means = np.nanmean(nan_array, axis=0)
for col in range(nan_array.shape[1]):
    mask = np.isnan(nan_array[:, col])
    nan_array[mask, col] = col_means[col]

nan_count_after = np.sum(np.isnan(nan_array))
print(f"    NaN count (after replacement)  : {nan_count_after}")

high_value = rev_array[rev_array > 2000]
print(f"\n  High-Value Transactions (>₹2000) : {len(high_value)} out of {N}")
print(f"  Average High-Value Amount         : ₹{np.mean(high_value):,.2f}")

sample_row  = [int(reshaped[0, 0]), int(reshaped[0, 1]), int(reshaped[0, 2])]
check_array = reshaped[:, :3].astype(int)
exists = any((check_array == sample_row).all(axis=1))
print(f"\n  Row {sample_row} exists in array : {exists}")

print("\n[STEP 9] Logistic Regression: Predicting Repeat Buyers")
print("-" * 50)

features = ["Age", "Total_Spent", "Num_Prev_Orders",
            "Avg_Rating", "Days_Since_Last", "Quantity", "Unit_Price"]
X = df[features].values
y = df["Repeat_Buyer"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

model = LogisticRegression(random_state=42, max_iter=500)
model.fit(X_train_s, y_train)

y_pred = model.predict(X_test_s)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec  = recall_score(y_test, y_pred, zero_division=0)
f1   = f1_score(y_test, y_pred, zero_division=0)
cm   = confusion_matrix(y_test, y_pred)

print(f"  Training Samples   : {len(X_train)}")
print(f"  Testing Samples    : {len(X_test)}")
print(f"\n  Model Evaluation:")
print(f"    Accuracy          : {acc:.4f}")
print(f"    Precision         : {prec:.4f}")
print(f"    Recall            : {rec:.4f}")
print(f"    F1-Score          : {f1:.4f}")
print("\n  Classification Report:")
print(classification_report(y_test, y_pred,
      target_names=["Non-Repeat", "Repeat"]))

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Non-Repeat", "Repeat"],
            yticklabels=["Non-Repeat", "Repeat"], ax=ax)
ax.set_title("Confusion Matrix – Repeat Buyer Prediction", fontweight="bold")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig("plot_5_confusion_matrix.png", dpi=120)
plt.close()
print("  Saved → plot_5_confusion_matrix.png")

print("\n[STEP 10] Standard Deviation & Aggregation Analysis")
print("-" * 50)

std_devs = df[numeric_cols].std().round(2)
print("  Standard Deviation per Feature:")
for col, val in std_devs.items():
    print(f"    {col:<20} : {val}")

threshold    = 500.0
pass_rate    = (df["Total_Spent"] > threshold).mean() * 100
print(f"\n  High-Value Customer Rate (Total_Spent > ₹{threshold}) : {pass_rate:.1f}%")

print("\n  Category-level Aggregation:")
cat_agg = df.groupby("Category").agg(
    Total_Revenue=("Revenue",    "sum"),
    Avg_Order    =("Revenue",    "mean"),
    Transactions =("Revenue",    "count"),
    Avg_Qty      =("Quantity",   "mean"),
    Std_Revenue  =("Revenue",    "std"),
).round(2)
print(cat_agg.to_string())

monthly_avg = df.groupby("Month")["Revenue"].mean()
fig, ax = plt.subplots(figsize=(12, 4))
monthly_avg.plot(kind="bar", color="mediumseagreen", edgecolor="black", ax=ax)
ax.set_title("Average Order Value per Month", fontsize=13, fontweight="bold")
ax.set_xlabel("Month")
ax.set_ylabel("Avg Revenue (₹)")
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig("plot_6_avg_monthly_aov.png", dpi=120)
plt.close()
print("\n  Saved → plot_6_avg_monthly_aov.png")

print("\n" + "=" * 65)
print("  RESULT")
print("=" * 65)
print("""
  The E-Commerce Customer Behavior Analysis was completed successfully.

  Key Findings:
  ─────────────────────────────────────────────────────────────
  • Dataset      : 500 transactions, 13 features, generated synthetically.
  • Revenue      : Total ₹{:,.0f}  |  Avg per order ₹{:,.0f}.
  • Top Category : {} with highest total revenue.
  • Z-Test       : {} (AOV vs ₹1200, α=0.05, Z={:.2f}).
  • ML Model     : Logistic Regression → Accuracy {:.1%}.
  • NaN Handling : {} NaNs introduced & replaced with column means.
  • Correlation  : Revenue strongly driven by Unit_Price.
  ─────────────────────────────────────────────────────────────

  Plots Saved:
    1. plot_1_correlation_heatmap.png
    2. plot_2_monthly_revenue.png
    3. plot_3_category_revenue.png
    4. plot_4_qty_vs_price.png
    5. plot_5_confusion_matrix.png
    6. plot_6_avg_monthly_aov.png
""".format(
    total_revenue, avg_order_value,
    cat_revenue.idxmax(),
    decision, z_score,
    acc,
    nan_count_before,
))