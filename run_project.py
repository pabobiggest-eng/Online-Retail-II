
# ================================================================
# ONLINE RETAIL II
# CUSTOMER COHORT RETENTION & REPEAT PURCHASE PREDICTION
# Full EDA + Cohort Analysis + Feature Engineering + 10 ML Models
# ================================================================

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.base import clone
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# ================================================================
# 1. CONFIGURATION
# ================================================================

DATA_PATH = Path("dataset/online_retail_II.xlsx")

OUTPUT = Path("outputs")
PROCESSED = OUTPUT / "processed"
REPORTS = OUTPUT / "reports"
FIGURES = OUTPUT / "figures"
MODELS_DIR = OUTPUT / "models"

for folder in [PROCESSED, REPORTS, FIGURES, MODELS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

print("=" * 75)
print("ONLINE RETAIL II: FULL DATA SCIENCE & MACHINE LEARNING PIPELINE")
print("=" * 75)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH.resolve()}\n"
        "Expected: dataset/online_retail_II.xlsx"
    )

# ================================================================
# 2. LOAD WORKBOOK AND CHECK DATA
# ================================================================

print("\n[1/8] LOADING DATA")

sheets = pd.read_excel(DATA_PATH, sheet_name=None)
frames = []

for sheet_name, part in sheets.items():
    part.columns = part.columns.astype(str).str.strip()
    part["SourceSheet"] = sheet_name
    frames.append(part)
    print(f"{sheet_name}: {len(part):,} rows")

raw = pd.concat(frames, ignore_index=True, sort=False)

required = [
    "Invoice", "StockCode", "Description", "Quantity",
    "InvoiceDate", "Price", "Customer ID", "Country"
]

missing_columns = [c for c in required if c not in raw.columns]
if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}\n"
        f"Available: {raw.columns.tolist()}"
    )

original_rows = len(raw)
original_duplicates = int(raw.duplicated().sum())

raw["InvoiceDate"] = pd.to_datetime(
    raw["InvoiceDate"], errors="coerce"
)
raw["Quantity"] = pd.to_numeric(raw["Quantity"], errors="coerce")
raw["Price"] = pd.to_numeric(raw["Price"], errors="coerce")

raw["IsCancelledInvoice"] = (
    raw["Invoice"].astype(str).str.strip().str.upper().str.startswith("C")
)

audit = {
    "original_rows": original_rows,
    "original_columns": len(raw.columns),
    "exact_duplicates": original_duplicates,
    "missing_values": {
        str(k): int(v) for k, v in raw.isna().sum().items()
    },
    "missing_customer_ids": int(raw["Customer ID"].isna().sum()),
    "missing_descriptions": int(raw["Description"].isna().sum()),
    "cancelled_invoice_rows": int(raw["IsCancelledInvoice"].sum()),
    "nonpositive_quantity_rows": int((raw["Quantity"] <= 0).sum()),
    "nonpositive_price_rows": int((raw["Price"] <= 0).sum()),
    "invalid_date_rows": int(raw["InvoiceDate"].isna().sum()),
    "source_sheets": list(sheets.keys()),
}

pd.DataFrame({
    "column": raw.columns,
    "dtype": [str(raw[c].dtype) for c in raw.columns],
}).to_csv(REPORTS / "column_information.csv", index=False)

raw.isna().sum().rename("missing_count").to_csv(
    REPORTS / "missing_values_before_cleaning.csv"
)

with open(REPORTS / "data_quality_before_cleaning.json", "w") as f:
    json.dump(audit, f, indent=4, default=str)

print("Original rows:", f"{original_rows:,}")
print("Exact duplicates:", f"{original_duplicates:,}")
print("Missing customer IDs:", f"{audit['missing_customer_ids']:,}")

# ================================================================
# 3. CLEANING
# ================================================================

print("\n[2/8] CLEANING TRANSACTIONS")

df = raw.drop(columns=["IsCancelledInvoice"]).copy()

# Exact duplicates removed only; original Excel file remains unchanged.
df = df.drop_duplicates().copy()
rows_after_deduplication = len(df)

# Exclude cancelled invoices from purchase analysis.
df = df.loc[
    ~df["Invoice"].astype(str).str.strip().str.upper().str.startswith("C")
].copy()

# Exclude records unsuitable for customer-level purchase modelling.
df = df.dropna(
    subset=["Invoice", "InvoiceDate", "Customer ID", "StockCode"]
).copy()

df = df.loc[
    (df["Quantity"] > 0) &
    (df["Price"] > 0)
].copy()

# Standardize fields.
df["Customer ID"] = (
    df["Customer ID"].astype(str).str.replace(r"\.0$", "", regex=True)
)
df["Invoice"] = df["Invoice"].astype(str).str.strip()
df["StockCode"] = df["StockCode"].astype(str).str.strip()
df["Description"] = df["Description"].fillna("Unknown product").astype(str)
df["Country"] = df["Country"].fillna("Unknown").astype(str)

df["Sales"] = df["Quantity"] * df["Price"]
df["InvoiceMonth"] = df["InvoiceDate"].dt.to_period("M")

if df.empty:
    raise ValueError("No usable transactions remain after cleaning.")

cleaning_report = {
    **audit,
    "rows_after_deduplication": int(rows_after_deduplication),
    "rows_after_cleaning": int(len(df)),
    "rows_removed_overall": int(original_rows - len(df)),
    "unique_customers": int(df["Customer ID"].nunique()),
    "unique_invoices": int(df["Invoice"].nunique()),
    "unique_products": int(df["StockCode"].nunique()),
    "date_start": str(df["InvoiceDate"].min()),
    "date_end": str(df["InvoiceDate"].max()),
}

with open(REPORTS / "data_quality_after_cleaning.json", "w") as f:
    json.dump(cleaning_report, f, indent=4, default=str)

df.to_csv(PROCESSED / "cleaned_transactions.csv", index=False)

df.isna().sum().rename("missing_count").to_csv(
    REPORTS / "missing_values_after_cleaning.csv"
)

print("Cleaned rows:", f"{len(df):,}")
print("Unique customers:", f"{df['Customer ID'].nunique():,}")
print("Unique invoices:", f"{df['Invoice'].nunique():,}")
print("Date range:", df["InvoiceDate"].min(), "to", df["InvoiceDate"].max())

# ================================================================
# 4. FULL EXPLORATORY DATA ANALYSIS
# ================================================================

print("\n[3/8] EXPLORATORY DATA ANALYSIS")

numeric_columns = ["Quantity", "Price", "Sales"]

statistics = df[numeric_columns].describe().T
statistics["skewness"] = df[numeric_columns].skew()
statistics["kurtosis"] = df[numeric_columns].kurtosis()
statistics.to_csv(REPORTS / "descriptive_statistics.csv")

# Outlier analysis: report outliers; do not blindly delete valid purchases.
outlier_records = []

for column in numeric_columns:
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outlier_records.append({
        "feature": column,
        "Q1": q1,
        "median": df[column].median(),
        "Q3": q3,
        "IQR": iqr,
        "lower_bound": lower,
        "upper_bound": upper,
        "outlier_count": int(
            ((df[column] < lower) | (df[column] > upper)).sum()
        ),
        "outlier_percentage": float(
            100 * ((df[column] < lower) | (df[column] > upper)).mean()
        ),
    })

pd.DataFrame(outlier_records).to_csv(
    REPORTS / "outlier_analysis.csv", index=False
)

# Revenue by month.
monthly = (
    df.groupby("InvoiceMonth", observed=True)
    .agg(
        revenue=("Sales", "sum"),
        orders=("Invoice", "nunique"),
        customers=("Customer ID", "nunique"),
        units=("Quantity", "sum"),
    )
    .reset_index()
)
monthly["InvoiceMonth"] = monthly["InvoiceMonth"].astype(str)
monthly.to_csv(REPORTS / "monthly_business_trends.csv", index=False)

plt.figure(figsize=(15, 6))
sns.lineplot(data=monthly, x="InvoiceMonth", y="revenue", marker="o")
plt.title("Monthly Revenue Trend")
plt.xticks(rotation=60)
plt.tight_layout()
plt.savefig(FIGURES / "monthly_revenue.png", dpi=160)
plt.close()

plt.figure(figsize=(15, 6))
sns.lineplot(data=monthly, x="InvoiceMonth", y="customers", marker="o")
plt.title("Monthly Unique Customers")
plt.xticks(rotation=60)
plt.tight_layout()
plt.savefig(FIGURES / "monthly_customers.png", dpi=160)
plt.close()

# Countries.
country_report = (
    df.groupby("Country")
    .agg(
        revenue=("Sales", "sum"),
        orders=("Invoice", "nunique"),
        customers=("Customer ID", "nunique"),
        units=("Quantity", "sum"),
    )
    .sort_values("revenue", ascending=False)
)
country_report.to_csv(REPORTS / "country_analysis.csv")

plt.figure(figsize=(12, 7))
sns.barplot(
    data=country_report.head(10).reset_index(),
    x="revenue",
    y="Country",
)
plt.title("Top 10 Countries by Revenue")
plt.tight_layout()
plt.savefig(FIGURES / "top_10_countries.png", dpi=160)
plt.close()

# Products.
product_report = (
    df.groupby(["StockCode", "Description"])
    .agg(
        revenue=("Sales", "sum"),
        units=("Quantity", "sum"),
        orders=("Invoice", "nunique"),
    )
    .sort_values("revenue", ascending=False)
    .reset_index()
)
product_report.to_csv(REPORTS / "product_analysis.csv", index=False)

top_products = product_report.head(15).copy()
top_products["product_label"] = (
    top_products["StockCode"] + " - " +
    top_products["Description"].str.slice(0, 35)
)

plt.figure(figsize=(13, 8))
sns.barplot(data=top_products, x="revenue", y="product_label")
plt.title("Top 15 Products by Revenue")
plt.tight_layout()
plt.savefig(FIGURES / "top_15_products.png", dpi=160)
plt.close()

# Numeric distributions: cap plots at the 99th percentile for readability.
for column in numeric_columns:
    upper = df[column].quantile(0.99)
    values = df.loc[df[column] <= upper, column]

    plt.figure(figsize=(9, 5))
    sns.histplot(values, bins=50, kde=True)
    plt.title(f"{column} Distribution (up to 99th percentile)")
    plt.tight_layout()
    plt.savefig(FIGURES / f"distribution_{column.lower()}.png", dpi=160)
    plt.close()

# Boxplots.
for column in numeric_columns:
    upper = df[column].quantile(0.99)

    plt.figure(figsize=(9, 4))
    sns.boxplot(x=df.loc[df[column] <= upper, column])
    plt.title(f"{column} Boxplot (up to 99th percentile)")
    plt.tight_layout()
    plt.savefig(FIGURES / f"boxplot_{column.lower()}.png", dpi=160)
    plt.close()

correlation = df[numeric_columns].corr()
correlation.to_csv(REPORTS / "numeric_correlation.csv")

plt.figure(figsize=(7, 5))
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Transaction Numeric Correlations")
plt.tight_layout()
plt.savefig(FIGURES / "numeric_correlation_heatmap.png", dpi=160)
plt.close()

# Customer purchase-frequency distribution.
orders_per_customer = df.groupby("Customer ID")["Invoice"].nunique()
orders_per_customer.rename("invoice_count").to_csv(
    REPORTS / "customer_invoice_frequency.csv"
)

plt.figure(figsize=(9, 5))
sns.histplot(orders_per_customer.clip(upper=orders_per_customer.quantile(0.99)),
             bins=40)
plt.title("Customer Invoice Frequency (99th percentile cap)")
plt.xlabel("Unique invoices per customer")
plt.tight_layout()
plt.savefig(FIGURES / "customer_invoice_frequency.png", dpi=160)
plt.close()

print("EDA tables and figures saved.")

# ================================================================
# 5. COHORT RETENTION ANALYSIS
# ================================================================

print("\n[4/8] CUSTOMER COHORT ANALYSIS")

cohort = df[["Customer ID", "Invoice", "InvoiceDate"]].copy()
cohort["InvoiceMonth"] = cohort["InvoiceDate"].dt.to_period("M")

first_purchase = cohort.groupby("Customer ID")["InvoiceDate"].min()
cohort["FirstPurchaseDate"] = cohort["Customer ID"].map(first_purchase)
cohort["CohortMonth"] = cohort["FirstPurchaseDate"].dt.to_period("M")

cohort["CohortIndex"] = (
    (cohort["InvoiceMonth"].dt.year - cohort["CohortMonth"].dt.year) * 12
    + cohort["InvoiceMonth"].dt.month - cohort["CohortMonth"].dt.month
    + 1
)

cohort_counts = (
    cohort.groupby(["CohortMonth", "CohortIndex"], observed=True)
    ["Customer ID"]
    .nunique()
    .reset_index(name="customers")
)

cohort_table = cohort_counts.pivot(
    index="CohortMonth",
    columns="CohortIndex",
    values="customers",
)

retention = cohort_table.div(cohort_table.iloc[:, 0], axis=0)

cohort_table.to_csv(REPORTS / "cohort_customer_counts.csv")
retention.to_csv(REPORTS / "cohort_retention_matrix.csv")

plt.figure(figsize=(16, 10))
sns.heatmap(retention, annot=True, fmt=".0%", cmap="YlGnBu")
plt.title("Monthly Customer Cohort Retention")
plt.xlabel("Months Since First Purchase")
plt.ylabel("First Purchase Cohort")
plt.tight_layout()
plt.savefig(FIGURES / "cohort_retention_heatmap.png", dpi=160)
plt.close()

# Overall repeat-customer summary.
customer_order_counts = df.groupby("Customer ID")["Invoice"].nunique()
repeat_customers = int((customer_order_counts >= 2).sum())
one_order_customers = int((customer_order_counts == 1).sum())

overall_repeat = pd.DataFrame([{
    "total_customers": int(len(customer_order_counts)),
    "repeat_customers": repeat_customers,
    "one_invoice_customers": one_order_customers,
    "repeat_customer_rate": (
        repeat_customers / len(customer_order_counts)
        if len(customer_order_counts) else np.nan
    ),
}])
overall_repeat.to_csv(REPORTS / "overall_repeat_purchase_summary.csv",
                      index=False)

print("Cohort retention matrix and heatmap saved.")

# ================================================================
# 6. CUSTOMER FEATURES AND LEAKAGE-SAFE TARGET
# ================================================================

print("\n[5/8] CUSTOMER FEATURE ENGINEERING")

# Target definition:
# First 90 days after first purchase = feature observation window.
# Days 90 to 180 = target window.
# Target 1 = at least one purchase during the target window.
# Only customers with full 180-day follow-up are eligible.

last_observation = df["InvoiceDate"].max()
customer_first_dates = df.groupby("Customer ID")["InvoiceDate"].min()

eligible_ids = customer_first_dates[
    customer_first_dates <= last_observation - pd.Timedelta(days=180)
].index

if len(eligible_ids) == 0:
    raise ValueError(
        "No eligible customers have 180 days of follow-up. "
        "Check the dataset date range."
    )

eligible = df[df["Customer ID"].isin(eligible_ids)].copy()
eligible["FirstPurchaseDate"] = eligible["Customer ID"].map(
    customer_first_dates
)

feature_end = eligible["FirstPurchaseDate"] + pd.Timedelta(days=90)
label_end = eligible["FirstPurchaseDate"] + pd.Timedelta(days=180)

feature_rows = eligible[
    eligible["InvoiceDate"] < feature_end
].copy()

target_rows = eligible[
    (eligible["InvoiceDate"] >= feature_end) &
    (eligible["InvoiceDate"] < label_end)
].copy()

# Core customer behaviour from the first 90 days.
features = feature_rows.groupby("Customer ID").agg(
    total_spend=("Sales", "sum"),
    total_quantity=("Quantity", "sum"),
    unique_products=("StockCode", "nunique"),
    unique_invoices=("Invoice", "nunique"),
    mean_line_price=("Price", "mean"),
    active_purchase_days=("InvoiceDate",
                          lambda s: s.dt.normalize().nunique()),
    last_purchase_date=("InvoiceDate", "max"),
    first_purchase_date=("FirstPurchaseDate", "first"),
)

features["recency_days_at_day_90"] = (
    features["first_purchase_date"] + pd.Timedelta(days=90)
    - features["last_purchase_date"]
).dt.total_seconds() / 86400

# Invoice-level basket features.
invoice_level = (
    feature_rows.groupby(["Customer ID", "Invoice"], as_index=False)
    .agg(
        basket_value=("Sales", "sum"),
        invoice_date=("InvoiceDate", "min"),
    )
)

basket_features = invoice_level.groupby("Customer ID").agg(
    average_basket_value=("basket_value", "mean"),
    maximum_basket_value=("basket_value", "max"),
)

features = features.join(basket_features)

features["items_per_invoice"] = (
    features["total_quantity"] /
    features["unique_invoices"].replace(0, np.nan)
)

def average_purchase_gap(series):
    dates = series.sort_values().drop_duplicates()
    gaps = dates.diff().dt.total_seconds().dropna() / 86400
    return float(gaps.mean()) if len(gaps) else np.nan

purchase_gaps = (
    invoice_level.groupby("Customer ID")["invoice_date"]
    .apply(average_purchase_gap)
    .rename("average_days_between_invoices")
)

features = features.join(purchase_gaps)

# Label using purchases only in the subsequent 90-day target window.
future_invoice_count = target_rows.groupby("Customer ID")["Invoice"].nunique()

features["repeat_purchase_next_90d"] = (
    features.index.to_series()
    .map(future_invoice_count)
    .fillna(0)
    .gt(0)
    .astype(int)
)

feature_columns = [
    "total_spend",
    "total_quantity",
    "unique_products",
    "unique_invoices",
    "mean_line_price",
    "active_purchase_days",
    "recency_days_at_day_90",
    "average_basket_value",
    "maximum_basket_value",
    "items_per_invoice",
    "average_days_between_invoices",
]

model_data = features[
    feature_columns + ["repeat_purchase_next_90d"]
].replace([np.inf, -np.inf], np.nan)

# Keep the customer first-purchase date separately for temporal splitting.
model_data["split_date"] = features["first_purchase_date"]
model_data = model_data.dropna(subset=["repeat_purchase_next_90d"])

model_data.to_csv(PROCESSED / "customer_ml_dataset.csv", index=True)

model_data[feature_columns].describe().T.to_csv(
    REPORTS / "customer_feature_statistics.csv"
)

class_counts = model_data["repeat_purchase_next_90d"].value_counts()
class_counts.rename_axis("target_class").to_csv(
    REPORTS / "target_class_distribution.csv"
)

print("Eligible customers:", len(model_data))
print("Target distribution:")
print(class_counts.to_string())

# ================================================================
# 7. TEN MACHINE LEARNING MODELS
# ================================================================

print("\n[6/8] MACHINE LEARNING MODEL COMPARISON")

X = model_data[feature_columns].copy()
y = model_data["repeat_purchase_next_90d"].astype(int)
dates = pd.to_datetime(model_data["split_date"])

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000, class_weight="balanced",
        random_state=RANDOM_STATE
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=8, class_weight="balanced",
        random_state=RANDOM_STATE
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=150, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1
    ),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=150, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1
    ),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Support Vector Machine": SVC(
        probability=True, class_weight="balanced",
        random_state=RANDOM_STATE
    ),
    "Gaussian Naive Bayes": GaussianNB(),
    "Gradient Boosting": GradientBoostingClassifier(
        random_state=RANDOM_STATE
    ),
    "Hist Gradient Boosting": HistGradientBoostingClassifier(
        random_state=RANDOM_STATE
    ),
    "Neural Network MLP": MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=350,
        early_stopping=True,
        random_state=RANDOM_STATE
    ),
}

# Prefer a temporal holdout: train on earlier first-purchase cohorts
# and test on later cohorts. Fall back to stratification only if needed.
sorted_dates = dates.sort_values()
date_cutoff = sorted_dates.iloc[int(0.80 * (len(sorted_dates) - 1))]

train_mask = dates < date_cutoff
test_mask = ~train_mask

X_train, X_test = X.loc[train_mask], X.loc[test_mask]
y_train, y_test = y.loc[train_mask], y.loc[test_mask]

split_type = "temporal"

if (
    len(X_train) == 0 or len(X_test) == 0
    or y_train.nunique() < 2 or y_test.nunique() < 2
    or y_train.value_counts().min() < 2
):
    print("Temporal split unsuitable; using stratified random holdout.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    split_type = "stratified random fallback"

if y_train.nunique() < 2:
    raise ValueError(
        "Training data has only one target class; ML cannot be trained."
    )

print("Split:", split_type)
print("Training customers:", len(X_train))
print("Test customers:", len(X_test))
print("Training class counts:")
print(y_train.value_counts().to_string())

results = []
fitted_models = {}

# Cross-validation only when each class has enough training examples.
minority_count = int(y_train.value_counts().min())
cv_splits = min(5, minority_count) if minority_count >= 2 else 0

for name, estimator in models.items():
    print(f"\nEvaluating: {name}")

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", clone(estimator)),
    ])

    cv_scores = []
    cv_mean = np.nan
    cv_std = np.nan

    # Cross-validation happens on training data only.
    # Preprocessing is inside the pipeline to avoid fold leakage.
    if cv_splits >= 2:
        cv = StratifiedKFold(
            n_splits=cv_splits,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
        try:
            cv_scores = cross_val_score(
                pipeline,
                X_train,
                y_train,
                scoring="average_precision",
                cv=cv,
                n_jobs=1,
                error_score=np.nan,
            )
            if np.isfinite(cv_scores).any():
                cv_mean = float(np.nanmean(cv_scores))
                cv_std = float(np.nanstd(cv_scores))
        except Exception as exc:
            print("Cross-validation unavailable:", str(exc)[:150])

    try:
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        probabilities = None
        if hasattr(pipeline, "predict_proba"):
            probabilities = pipeline.predict_proba(X_test)[:, 1]
        elif hasattr(pipeline, "decision_function"):
            raw_scores = pipeline.decision_function(X_test)
            probabilities = (
                raw_scores - raw_scores.min()
            ) / (raw_scores.max() - raw_scores.min() + 1e-12)

        roc_auc = np.nan
        pr_auc = np.nan

        if probabilities is not None and y_test.nunique() == 2:
            roc_auc = roc_auc_score(y_test, probabilities)
            pr_auc = average_precision_score(y_test, probabilities)

        row = {
            "model": name,
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "f1_score": f1_score(
                y_test, predictions, zero_division=0
            ),
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "cv_pr_auc_mean": cv_mean,
            "cv_pr_auc_std": cv_std,
            "status": "success",
        }

        results.append(row)
        fitted_models[name] = pipeline

        print(
            "F1:", round(row["f1_score"], 4),
            "| PR-AUC:", (
                round(pr_auc, 4) if np.isfinite(pr_auc) else "N/A"
            ),
        )

    except Exception as exc:
        print("Model failed:", name, str(exc)[:200])
        results.append({
            "model": name,
            "accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1_score": np.nan,
            "roc_auc": np.nan,
            "pr_auc": np.nan,
            "cv_pr_auc_mean": cv_mean,
            "cv_pr_auc_std": cv_std,
            "status": f"failed: {str(exc)[:150]}",
        })

results_df = pd.DataFrame(results)
results_df.to_csv(REPORTS / "model_comparison.csv", index=False)

successful = results_df[
    results_df["status"] == "success"
].copy()

if successful.empty:
    raise RuntimeError(
        "All models failed. Review target_class_distribution.csv "
        "and the terminal output."
    )

# Prefer cross-validated PR-AUC to choose a model when available.
if successful["cv_pr_auc_mean"].notna().any():
    ranked = successful.sort_values(
        ["cv_pr_auc_mean", "f1_score"],
        ascending=False,
        na_position="last",
    )
    selection_metric = "cv_pr_auc_mean"
else:
    ranked = successful.sort_values(
        ["pr_auc", "f1_score"],
        ascending=False,
        na_position="last",
    )
    selection_metric = "pr_auc_then_f1"

best_name = ranked.iloc[0]["model"]
best_model = fitted_models[best_name]

joblib.dump(best_model, MODELS_DIR / "best_repeat_purchase_model.joblib")

# ================================================================
# 8. FINAL EVALUATION AND MODEL REPORT
# ================================================================

print("\n[7/8] FINAL MODEL EVALUATION")

final_predictions = best_model.predict(X_test)

if hasattr(best_model, "predict_proba"):
    final_probabilities = best_model.predict_proba(X_test)[:, 1]
else:
    scores = best_model.decision_function(X_test)
    final_probabilities = (
        scores - scores.min()
    ) / (scores.max() - scores.min() + 1e-12)

cm = confusion_matrix(y_test, final_predictions, labels=[0, 1])

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title(f"Confusion Matrix: {best_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(FIGURES / "best_model_confusion_matrix.png", dpi=160)
plt.close()

report = classification_report(
    y_test, final_predictions, zero_division=0
)

with open(REPORTS / "best_model_classification_report.txt", "w",
          encoding="utf-8") as f:
    f.write(f"Best model: {best_name}\n")
    f.write(f"Selection approach: {selection_metric}\n")
    f.write(f"Split type: {split_type}\n")
    f.write(f"Train customers: {len(X_train)}\n")
    f.write(f"Test customers: {len(X_test)}\n\n")
    f.write(report)

# Model comparison chart.
plot_df = successful.sort_values("f1_score", ascending=True)

plt.figure(figsize=(11, 7))
sns.barplot(data=plot_df, x="f1_score", y="model")
plt.title("Model Comparison by F1 Score")
plt.xlim(0, 1)
plt.tight_layout()
plt.savefig(FIGURES / "model_comparison_f1.png", dpi=160)
plt.close()

# Tree-based feature importance where supported.
final_estimator = best_model.named_steps["classifier"]

if hasattr(final_estimator, "feature_importances_"):
    importance = pd.DataFrame({
        "feature": feature_columns,
        "importance": final_estimator.feature_importances_,
    }).sort_values("importance", ascending=False)

    importance.to_csv(
        REPORTS / "feature_importance.csv", index=False
    )

    plt.figure(figsize=(10, 7))
    sns.barplot(data=importance, x="importance", y="feature")
    plt.title(f"Feature Importance: {best_name}")
    plt.tight_layout()
    plt.savefig(FIGURES / "feature_importance.png", dpi=160)
    plt.close()

summary = {
    "dataset": str(DATA_PATH),
    "original_rows": int(original_rows),
    "cleaned_rows": int(len(df)),
    "cleaned_customers": int(df["Customer ID"].nunique()),
    "ml_eligible_customers": int(len(model_data)),
    "feature_count": int(len(feature_columns)),
    "target": "purchase in days 90-180 after first purchase",
    "feature_window": "first 90 days after first purchase",
    "split_type": split_type,
    "best_model": best_name,
    "selection_approach": selection_metric,
    "successful_models": int(len(successful)),
    "model_results": results_df.replace(
        {np.nan: None}
    ).to_dict(orient="records"),
}

with open(REPORTS / "project_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4, default=str)

print("\nBest model:", best_name)
print("Selection method:", selection_metric)
print("\nClassification report:\n", report)

# ================================================================
# FINAL
# ================================================================

print("\n[8/8] PIPELINE COMPLETED")
print("=" * 75)
print("Cleaned data:", PROCESSED / "cleaned_transactions.csv")
print("ML dataset:", PROCESSED / "customer_ml_dataset.csv")
print("Reports folder:", REPORTS.resolve())
print("Figures folder:", FIGURES.resolve())
print("Saved model:", MODELS_DIR / "best_repeat_purchase_model.joblib")
print("Model comparison:", REPORTS / "model_comparison.csv")
print("\nGitHub push will be done AFTER inspecting all outputs.")
print("=" * 75)
