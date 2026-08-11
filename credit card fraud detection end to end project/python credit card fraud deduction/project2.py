import os
import textwrap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, average_precision_score, classification_report, confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

file_path = r"credit_card_fraud_2026.csv"
model_path = r"fraud_model.joblib"


def find_label_column(df):
    for col in ["Class", "is_fraud", "fraud", "fraud_label"]:
        if col in df.columns:
            return col
    return None


def find_amount_column(df):
    for col in df.columns:
        if col.lower() in {"amount", "amount_usd", "transaction_amount"}:
            return col
    for col in df.columns:
        if "amount" in col.lower():
            return col
    return None


def clean_data(df):
    print("\nData cleaning:")
    initial_len = len(df)

    missing = df.isna().sum()
    missing_cols = missing[missing > 0]
    if not missing_cols.empty:
        print("Missing values found:")
        print(missing_cols)
        for col in missing_cols.index:
            if df[col].dtype == "bool":
                df[col] = df[col].fillna(False)
            elif np.issubdtype(df[col].dtype, np.number):
                median = df[col].median()
                df[col] = df[col].fillna(median)
            else:
                mode = df[col].mode(dropna=True)
                if not mode.empty:
                    df[col] = df[col].fillna(mode[0])
                else:
                    df[col] = df[col].fillna("")
        print("Filled missing values: numeric -> median, categorical -> mode.")
    else:
        print("No missing values found.")

    dup_count = df.duplicated().sum()
    if dup_count > 0:
        print(f"Found {dup_count} duplicate rows. Dropping duplicates.")
        df = df.drop_duplicates().reset_index(drop=True)
    else:
        print("No duplicate rows found.")

    object_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    for col in object_cols:
        coerced = pd.to_numeric(df[col].astype(str).str.replace(",", "", regex=False), errors="coerce")
        num_coerced = coerced.notna().sum()
        if num_coerced >= len(df) * 0.9:
            print(f"Converting object column '{col}' to numeric.")
            df[col] = coerced

    amount_col = find_amount_column(df)
    if amount_col is not None:
        negative_count = (df[amount_col] < 0).sum()
        if negative_count > 0:
            print(f"Found {negative_count} negative values in '{amount_col}'. Removing those rows.")
            df = df[df[amount_col] >= 0].reset_index(drop=True)
        zero_count = (df[amount_col] == 0).sum()
        if zero_count > 0:
            print(f"Found {zero_count} zero values in '{amount_col}'. Keeping them but please verify if valid.")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    outlier_report = {}
    for col in numeric_cols:
        if df[col].nunique() < 10:
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = ((df[col] < lower) | (df[col] > upper)).sum()
        if count > 0:
            outlier_report[col] = count

    if outlier_report:
        print("\nOutlier detection (IQR) counts per numeric column:")
        for col, count in outlier_report.items():
            print(f"  {col}: {count} outliers")
    else:
        print("No numeric outliers found by IQR.")

    final_len = len(df)
    print(f"Rows before cleaning: {initial_len}, after cleaning: {final_len}")
    return df


def create_features(df):
    print("\nFeature engineering:")
    if "amount_usd" in df.columns:
        df["amount_log"] = np.log1p(df["amount_usd"])
        df["high_amount"] = df["amount_usd"] > df["amount_usd"].quantile(0.95)

    if "account_balance_usd" in df.columns and "amount_usd" in df.columns:
        df["balance_ratio"] = df["amount_usd"] / (df["account_balance_usd"] + 1)
        df["low_balance_flag"] = df["account_balance_usd"] < df["amount_usd"]

    if "time_of_day_hour" in df.columns:
        df["is_night"] = df["time_of_day_hour"].between(0, 5) | df["time_of_day_hour"].between(22, 23)
        df["is_peak_hour"] = df["time_of_day_hour"].between(16, 20)

    if "day_of_week" in df.columns:
        df["is_weekend"] = df["day_of_week"].isin([5, 6, 7])

    if "velocity_score" in df.columns:
        df["high_velocity"] = df["velocity_score"] > df["velocity_score"].quantile(0.75)

    if "merchant_risk_score" in df.columns:
        df["high_merchant_risk"] = df["merchant_risk_score"] > df["merchant_risk_score"].median()

    if "prior_disputes" in df.columns:
        df["has_prior_disputes"] = df["prior_disputes"] > 0

    boolean_pairs = [
        ("is_foreign_transaction", "used_vpn", "foreign_and_vpn"),
        ("is_new_merchant", "high_merchant_risk", "new_high_risk_merchant"),
        ("ip_country_mismatch", "billing_shipping_mismatch", "mismatch_pair"),
    ]
    for col1, col2, new_col in boolean_pairs:
        if col1 in df.columns and col2 in df.columns:
            df[new_col] = df[col1].astype(bool) & df[col2].astype(bool)

    for cat_col in ["merchant_category", "channel", "card_type", "device_type"]:
        if cat_col in df.columns:
            freq_col = f"{cat_col}_freq"
            df[freq_col] = df.groupby(cat_col)[cat_col].transform("count")

    if "customer_age" in df.columns:
        df["age_bucket"] = pd.cut(df["customer_age"], bins=[0, 25, 35, 50, 65, 100], labels=["<25", "25-35", "35-50", "50-65", ">65"], include_lowest=True)

    engineered_cols = [c for c in df.columns if c not in ["transaction_id", "is_fraud", "Class"] and c.endswith(("_log", "_flag", "_ratio", "_risk", "_rate", "_bucket", "_freq", "_amount"))]
    print(f"Created {len(engineered_cols)} engineered feature(s): {engineered_cols[:10]}{'...' if len(engineered_cols) > 10 else ''}")
    return df


def compute_business_kpis(df, label_col, amount_col):
    print("\nBusiness KPIs:")
    total_txns = len(df)
    print(f"Total transactions: {total_txns}")

    fraud_txns = int(df[label_col].sum()) if label_col in df.columns else 0
    fraud_rate = fraud_txns / total_txns if total_txns else 0
    print(f"Fraud transactions: {fraud_txns}")
    print(f"Fraud rate: {fraud_rate:.4f} ({fraud_rate*100:.2f}%)")

    kpi_info = {
        "total_transactions": total_txns,
        "fraud_transactions": fraud_txns,
        "fraud_rate": fraud_rate,
    }

    if amount_col is not None:
        total_volume = df[amount_col].sum()
        avg_amount = df[amount_col].mean()
        fraud_volume = df.loc[df[label_col] == 1, amount_col].sum()
        nonfraud_volume = df.loc[df[label_col] == 0, amount_col].sum()
        avg_fraud_amount = df.loc[df[label_col] == 1, amount_col].mean()
        avg_nonfraud_amount = df.loc[df[label_col] == 0, amount_col].mean()
        volume_share = fraud_volume / total_volume if total_volume else 0

        print(f"Total transaction volume: {total_volume:,.2f}")
        print(f"Average transaction amount: {avg_amount:,.2f}")
        print(f"Total fraud volume: {fraud_volume:,.2f}")
        print(f"Average fraud transaction amount: {avg_fraud_amount:,.2f}")
        print(f"Average non-fraud transaction amount: {avg_nonfraud_amount:,.2f}")
        print(f"Fraud volume share: {volume_share:.4f} ({volume_share*100:.2f}%)")

        kpi_info.update({
            "total_volume": total_volume,
            "avg_amount": avg_amount,
            "fraud_volume": fraud_volume,
            "avg_fraud_amount": avg_fraud_amount,
            "avg_nonfraud_amount": avg_nonfraud_amount,
            "volume_share": volume_share,
        })

    if "velocity_score" in df.columns:
        avg_velocity_fraud = df.loc[df[label_col] == 1, "velocity_score"].mean()
        avg_velocity_nonfraud = df.loc[df[label_col] == 0, "velocity_score"].mean()
        print(f"Average velocity (fraud): {avg_velocity_fraud:.2f}")
        print(f"Average velocity (non-fraud): {avg_velocity_nonfraud:.2f}")
        kpi_info["avg_velocity_fraud"] = avg_velocity_fraud
        kpi_info["avg_velocity_nonfraud"] = avg_velocity_nonfraud

    if "merchant_risk_score" in df.columns:
        avg_merchant_risk_fraud = df.loc[df[label_col] == 1, "merchant_risk_score"].mean()
        avg_merchant_risk_nonfraud = df.loc[df[label_col] == 0, "merchant_risk_score"].mean()
        print(f"Average merchant risk (fraud): {avg_merchant_risk_fraud:.2f}")
        print(f"Average merchant risk (non-fraud): {avg_merchant_risk_nonfraud:.2f}")
        kpi_info["avg_merchant_risk_fraud"] = avg_merchant_risk_fraud
        kpi_info["avg_merchant_risk_nonfraud"] = avg_merchant_risk_nonfraud

    if "is_new_merchant" in df.columns:
        new_merch_rate = df.loc[df['is_new_merchant'] == True, label_col].mean()
        existing_rate = df.loc[df['is_new_merchant'] == False, label_col].mean()
        print(f"Fraud rate for new merchants: {new_merch_rate:.4f}")
        print(f"Fraud rate for existing merchants: {existing_rate:.4f}")
        kpi_info["new_merchant_fraud_rate"] = new_merch_rate
        kpi_info["existing_merchant_fraud_rate"] = existing_rate

    if "is_foreign_transaction" in df.columns:
        foreign_rate = df.loc[df['is_foreign_transaction'] == True, label_col].mean()
        domestic_rate = df.loc[df['is_foreign_transaction'] == False, label_col].mean()
        print(f"Fraud rate for foreign transactions: {foreign_rate:.4f}")
        print(f"Fraud rate for domestic transactions: {domestic_rate:.4f}")
        kpi_info["foreign_transaction_fraud_rate"] = foreign_rate
        kpi_info["domestic_transaction_fraud_rate"] = domestic_rate

    if "device_type" in df.columns and "card_type" in df.columns:
        fraud_by_card = df.groupby('card_type')[label_col].mean().sort_values(ascending=False).head(5)
        print("Top card types by fraud rate:")
        print(fraud_by_card.to_string())
        kpi_info["top_card_types_by_fraud_rate"] = fraud_by_card.to_dict()

    if "merchant_category" in df.columns:
        fraud_by_merchant = df.groupby('merchant_category')[label_col].mean().sort_values(ascending=False).head(5)
        print("Top merchant categories by fraud rate:")
        print(fraud_by_merchant.to_string())
        kpi_info["top_merchant_categories_by_fraud_rate"] = fraud_by_merchant.to_dict()

    return kpi_info


def plot_summary(df, label_col, amount_col):
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.countplot(x=label_col, data=df, color="#4c72b0")
    plt.title(f"{label_col} counts")
    plt.xlabel(label_col)
    plt.ylabel("Count")

    plt.subplot(1, 2, 2)
    if amount_col is not None:
        sns.boxplot(x=label_col, y=amount_col, data=df, color="#4c72b0")
        plt.title(f"{amount_col} by {label_col}")
        plt.xlabel(label_col)
        plt.ylabel(amount_col)
    else:
        plt.text(0.5, 0.5, "No amount column found", ha="center", va="center")
        plt.axis("off")

    plt.tight_layout()
    plot_path = "fraud_summary_plots.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"\nSaved plot to {plot_path}")


def find_country_column(df):
    for col in df.columns:
        if "country" in col.lower():
            return col
    return None


def find_datetime_column(df):
    candidates = [col for col in df.columns if any(token in col.lower() for token in ["date", "time", "timestamp", "datetime", "txn"])]
    for col in candidates:
        coerced = pd.to_datetime(df[col], errors="coerce")
        if coerced.notna().sum() >= len(df) * 0.8:
            return col
    return None


def print_feature_importance(model, X_train):
    print("\nFeature importance:")
    rf = model.named_steps["classifier"]
    ct = model.named_steps["preprocessor"]

    numeric_features = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

    feature_names = []
    if numeric_features:
        feature_names.extend(numeric_features)
    if categorical_features:
        cat_features = ct.named_transformers_["cat"].get_feature_names_out(categorical_features)
        feature_names.extend(cat_features.tolist())

    importances = rf.feature_importances_
    importance_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    importance_df = importance_df.sort_values(by="importance", ascending=False).reset_index(drop=True)
    print(importance_df.head(15).to_string(index=False))

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="importance", y="feature", data=importance_df.head(20), color="#4c72b0", ax=ax)
    ax.set_title("Top 20 Feature Importances")
    save_figure(fig, "feature_importances.png")
    return importance_df


def save_figure(fig, filename, directory="eda_plots"):
    if directory and directory != ".":
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, filename)
    else:
        path = filename
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {path}")


def save_cleaned_data(df, filename="cleaned_data.csv"):
    df.to_csv(filename, index=False)
    print(f"Saved cleaned data: {filename}")


def save_confusion_matrix(cm, filename="confusion_matrix.png"):
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    save_figure(fig, filename, directory=".")


def build_report_text(kpi_info, metrics, importance_df):
    lines = [
        "Business and model summary",
        "==========================",
        f"Total transactions: {kpi_info.get('total_transactions', 0):,}",
        f"Fraud transactions: {kpi_info.get('fraud_transactions', 0):,}",
        f"Fraud rate: {kpi_info.get('fraud_rate', 0):.4f} ({kpi_info.get('fraud_rate', 0) * 100:.2f}%)",
    ]

    if "total_volume" in kpi_info:
        lines += [
            f"Total transaction volume: {kpi_info.get('total_volume', 0):,.2f}",
            f"Average transaction amount: {kpi_info.get('avg_amount', 0):,.2f}",
            f"Total fraud volume: {kpi_info.get('fraud_volume', 0):,.2f}",
            f"Average fraud transaction amount: {kpi_info.get('avg_fraud_amount', 0):,.2f}",
            f"Average non-fraud transaction amount: {kpi_info.get('avg_nonfraud_amount', 0):,.2f}",
            f"Fraud volume share: {kpi_info.get('volume_share', 0):.4f} ({kpi_info.get('volume_share', 0) * 100:.2f}%)",
        ]

    if "avg_velocity_fraud" in kpi_info:
        lines += [
            f"Average velocity (fraud): {kpi_info.get('avg_velocity_fraud', 0):.2f}",
            f"Average velocity (non-fraud): {kpi_info.get('avg_velocity_nonfraud', 0):.2f}",
        ]

    if "avg_merchant_risk_fraud" in kpi_info:
        lines += [
            f"Average merchant risk (fraud): {kpi_info.get('avg_merchant_risk_fraud', 0):.2f}",
            f"Average merchant risk (non-fraud): {kpi_info.get('avg_merchant_risk_nonfraud', 0):.2f}",
        ]

    if "new_merchant_fraud_rate" in kpi_info:
        lines += [
            f"Fraud rate for new merchants: {kpi_info.get('new_merchant_fraud_rate', 0):.4f}",
            f"Fraud rate for existing merchants: {kpi_info.get('existing_merchant_fraud_rate', 0):.4f}",
        ]

    if "foreign_transaction_fraud_rate" in kpi_info:
        lines += [
            f"Fraud rate for foreign transactions: {kpi_info.get('foreign_transaction_fraud_rate', 0):.4f}",
            f"Fraud rate for domestic transactions: {kpi_info.get('domestic_transaction_fraud_rate', 0):.4f}",
        ]

    if "top_card_types_by_fraud_rate" in kpi_info:
        lines.append("\nTop card types by fraud rate:")
        for card_type, rate in kpi_info["top_card_types_by_fraud_rate"].items():
            lines.append(f"  {card_type}: {rate:.4f}")

    if "top_merchant_categories_by_fraud_rate" in kpi_info:
        lines.append("\nTop merchant categories by fraud rate:")
        for merchant_category, rate in kpi_info["top_merchant_categories_by_fraud_rate"].items():
            lines.append(f"  {merchant_category}: {rate:.4f}")

    lines += [
        "\nModel evaluation",
        "----------------",
        f"Accuracy: {metrics.get('accuracy', 0):.4f}",
        f"ROC AUC: {metrics.get('roc_auc', 0):.4f}",
        f"Average precision: {metrics.get('avg_precision', 0):.4f}",
        "\nClassification report:",
        metrics.get('classification_report', ""),
        "\nConfusion matrix:",
        np.array2string(metrics.get('confusion_matrix', np.array([[]])), separator=', '),
    ]

    if importance_df is not None:
        lines.append("\nTop feature importances:")
        lines.append(importance_df.head(10).to_string(index=False))

    return "\n".join(lines)


def save_business_report(report_text, filename):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"Saved business report: {filename}")


def save_summary_pdf(report_text, filename):
    with PdfPages(filename) as pdf:
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")
        y = 0.95
        for raw_line in report_text.splitlines():
            wrapped_lines = textwrap.wrap(raw_line, width=90) or [""]
            for line in wrapped_lines:
                if y < 0.05:
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)
                    fig, ax = plt.subplots(figsize=(8.27, 11.69))
                    ax.axis("off")
                    y = 0.95
                ax.text(0.01, y, line, fontsize=10, family="monospace")
                y -= 0.03
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
    print(f"Saved summary PDF: {filename}")


def save_feature_importance_csv(importance_df, filename):
    importance_df.to_csv(filename, index=False)
    print(f"Saved feature importance CSV: {filename}")


def save_predictions_csv(model, X, df, label_col, amount_col, filename):
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= 0.5).astype(int)
    output = pd.DataFrame({
        "predicted_label": preds,
        "fraud_probability": proba,
    })
    if label_col in df.columns:
        output["actual_label"] = df[label_col].astype(int).values
    if "transaction_id" in df.columns:
        output.insert(0, "transaction_id", df["transaction_id"].values)
    else:
        output.insert(0, "row_id", df.index.values)
    if amount_col is not None:
        output[amount_col] = df[amount_col].values

    output.to_csv(filename, index=False)
    print(f"Saved predictions CSV: {filename}")


def tune_hyperparameters(X, y, pipeline):
    print("\nHyperparameter tuning:")
    param_grid = {
        "classifier__n_estimators": [100],
        "classifier__max_depth": [20],
        "classifier__max_features": ["sqrt"],
        "classifier__min_samples_split": [2],
    }
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=42),
        n_jobs=-1,
        verbose=0,
        refit=True,
    )
    search.fit(X, y)
    print(f"Best hyperparameters: {search.best_params_}")
    print(f"Best cross-validated ROC AUC: {search.best_score_:.4f}")
    return search.best_estimator_


def make_eda_plots(df, label_col, amount_col):
    if label_col is None:
        print("\nSkipping EDA plots because no label column was found.")
        return

    country_col = find_country_column(df)
    datetime_col = find_datetime_column(df)
    top_n = 12

    os.makedirs("eda_plots", exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x=label_col, data=df, color="#4c72b0", ax=ax)
    ax.set_title("Fraud Distribution")
    ax.set_ylabel("Count")
    save_figure(fig, "fraud_distribution_count.png")

    fig, ax = plt.subplots(figsize=(6, 6))
    counts = df[label_col].value_counts().sort_index()
    labels = [str(x) for x in counts.index.tolist()]
    ax.pie(counts, labels=labels, autopct="%1.1f%%", startangle=90, colors=sns.color_palette("pastel", len(counts)))
    ax.set_title("Fraud Distribution (Pie)")
    save_figure(fig, "fraud_distribution_pie.png")

    if amount_col is not None:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df[amount_col], bins=50, color="#4c72b0", ax=ax)
        ax.set_title("Transaction Amount Histogram")
        ax.set_xlabel(amount_col)
        save_figure(fig, "amount_histogram.png")

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(data=df, x=amount_col, hue=label_col, bins=40, alpha=0.6, element="step", ax=ax)
        ax.set_title("Transaction Amount Distribution by Fraud")
        ax.set_xlabel(amount_col)
        save_figure(fig, "amount_histogram_by_fraud.png")

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(x=label_col, y=amount_col, data=df, color="#4c72b0", ax=ax)
        ax.set_title("Transaction Amount by Fraud")
        save_figure(fig, "amount_boxplot_by_fraud.png")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr, cmap="coolwarm", center=0, square=True, linewidths=0.5, ax=ax)
        ax.set_title("Correlation Heatmap")
        save_figure(fig, "correlation_heatmap.png", directory=".")

    category_cols = [
        ("merchant_category", "Merchant Category"),
        ("channel", "Transaction Channel"),
        ("card_type", "Card Type"),
        ("device_type", "Device Type"),
        ("auth_method", "Auth Method"),
        ("day_of_week", "Day of Week"),
    ]

    for col, label in category_cols:
        if col in df.columns:
            top_vals = df[col].value_counts().nlargest(top_n)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=top_vals.values, y=top_vals.index, color="#4c72b0", ax=ax)
            ax.set_title(f"Top {len(top_vals)} {label} Values")
            ax.set_xlabel("Count")
            ax.set_ylabel(label)
            save_figure(fig, f"{col}_counts.png")

            fig, ax = plt.subplots(figsize=(10, 6))
            rates = df.groupby(col)[label_col].mean().loc[top_vals.index]
            sns.barplot(x=rates.values, y=rates.index, color="#ff7f0e", ax=ax)
            ax.set_title(f"Fraud Rate by {label}")
            ax.set_xlabel("Fraud Rate")
            ax.set_ylabel(label)
            save_figure(fig, f"{col}_fraud_rate.png")

    if "time_of_day_hour" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        order = list(range(24))
        sns.countplot(x="time_of_day_hour", data=df, order=order, color="#4c72b0", ax=ax)
        ax.set_title("Transactions by Hour of Day")
        ax.set_xlabel("Hour")
        ax.set_ylabel("Count")
        save_figure(fig, "hourly_transactions.png")

    if "day_of_week" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        order = sorted(df["day_of_week"].dropna().unique())
        sns.countplot(x="day_of_week", data=df, order=order, color="#4c72b0", ax=ax)
        ax.set_title("Transactions by Day of Week")
        ax.set_xlabel("Day of Week")
        ax.set_ylabel("Count")
        save_figure(fig, "weekday_transactions.png")

    if country_col is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        top_vals = df[country_col].value_counts().nlargest(top_n)
        sns.barplot(x=top_vals.values, y=top_vals.index, color="#4c72b0", ax=ax)
        ax.set_title("Top Countries")
        ax.set_xlabel("Count")
        ax.set_ylabel(country_col)
        save_figure(fig, "country_counts.png")

        fig, ax = plt.subplots(figsize=(10, 6))
        rates = df.groupby(country_col)[label_col].mean().loc[top_vals.index]
        sns.barplot(x=rates.values, y=rates.index, color="#ff7f0e", ax=ax)
        ax.set_title("Fraud Rate by Country")
        ax.set_xlabel("Fraud Rate")
        ax.set_ylabel(country_col)
        save_figure(fig, "country_fraud_rate.png")
    else:
        print("No country column found for country charts.")

    if datetime_col is not None:
        df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
        if df[datetime_col].notna().sum() >= len(df) * 0.8:
            monthly = df.set_index(datetime_col).resample("ME")[label_col].mean()
            fig, ax = plt.subplots(figsize=(10, 5))
            monthly.plot(ax=ax, marker="o")
            ax.set_title("Monthly Fraud Rate Trend")
            ax.set_xlabel("Month")
            ax.set_ylabel("Fraud Rate")
            save_figure(fig, "monthly_fraud_trend.png")
        else:
            print("Datetime column found but not enough valid entries for monthly trend.")
    else:
        print("No datetime column found for monthly fraud trend.")

    if "merchant_risk_score" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df["merchant_risk_score"], bins=40, color="#4c72b0", ax=ax)
        ax.set_title("Merchant Risk Score Distribution")
        save_figure(fig, "merchant_risk_score_histogram.png")

    if "customer_age" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df["customer_age"], bins=30, kde=True, color="#4c72b0", ax=ax)
        ax.set_title("Customer Age Distribution")
        save_figure(fig, "customer_age_distribution.png")

    if "velocity_score" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df["velocity_score"], bins=40, kde=True, color="#4c72b0", ax=ax)
        ax.set_title("Velocity Score Distribution")
        save_figure(fig, "velocity_score_distribution.png")

    if "account_balance_usd" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df["account_balance_usd"], bins=40, kde=True, color="#4c72b0", ax=ax)
        ax.set_title("Account Balance Distribution")
        save_figure(fig, "account_balance_distribution.png")


def build_and_evaluate_model(df, label_col, amount_col):
    X = df.drop(columns=[label_col])
    y = df[label_col]

    if "transaction_id" in X.columns:
        X = X.drop(columns=["transaction_id"])

    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()

    if not numeric_features and not categorical_features:
        raise ValueError("No valid features found for modeling.")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ],
        remainder="drop",
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=42, n_jobs=-1, n_estimators=100)),
        ]
    )

    tuned_model = tune_hyperparameters(X, y, model)

    print("\nRunning stratified cross-validation with tuned model...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_proba = cross_val_predict(tuned_model, X, y, cv=skf, method="predict_proba")[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    roc_auc = roc_auc_score(y, y_proba)
    fpr, tpr, _ = roc_curve(y, y_proba)
    precision, recall, _ = precision_recall_curve(y, y_proba)
    avg_precision = average_precision_score(y, y_proba)
    cm = confusion_matrix(y, y_pred)

    print("\nCross-validation evaluation")
    print("Accuracy:", round(accuracy_score(y, y_pred), 4))
    print("ROC AUC:", round(roc_auc, 4))
    print("Average precision:", round(avg_precision, 4))
    print("\nClassification report:")
    print(classification_report(y, y_pred, zero_division=0))
    print("\nConfusion matrix:")
    print(cm)

    save_confusion_matrix(cm, "confusion_matrix.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})", color="#1f77b4")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    save_figure(fig, "roc_curve.png", directory=".")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, label=f"Precision-Recall (AP = {avg_precision:.4f})", color="#d62728")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    save_figure(fig, "precision_recall_curve.png")

    print("\nTraining final model on full dataset...")
    model.fit(X, y)
    importance_df = print_feature_importance(model, X)
    save_feature_importance_csv(importance_df, "feature_importance.csv")
    save_predictions_csv(model, X, df, label_col, amount_col, "predictions.csv")

    dump(model, model_path)
    print(f"\nSaved trained model to {model_path}")

    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "roc_auc": roc_auc,
        "avg_precision": avg_precision,
        "classification_report": classification_report(y, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y, y_pred),
    }
    return model, metrics, importance_df


def main():
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    df = pd.read_csv(file_path)
    print("Loaded data:", df.shape)

    df = clean_data(df)
    df = create_features(df)
    print("\nColumns after cleaning and feature engineering:", list(df.columns))
    print("\nFirst 5 rows after cleaning and feature engineering:")
    print(df.head().to_string(index=False))

    print("\nInfo after cleaning and feature engineering:")
    df.info()

    label_col = find_label_column(df)
    amount_col = find_amount_column(df)

    save_cleaned_data(df, "cleaned_data.csv")

    if label_col is not None:
        kpi_info = compute_business_kpis(df, label_col, amount_col)
        print(f"\n{label_col} value counts:")
        print(df[label_col].value_counts(dropna=False))
    else:
        kpi_info = {}
        print("\nNo fraud label column found.")

    if amount_col is not None:
        print(f"\n{amount_col} summary:")
        print(df[amount_col].describe())

        if label_col is not None:
            print(f"\n{amount_col} summary by {label_col}:")
            print(df.groupby(label_col)[amount_col].describe())
    else:
        print("\nNo amount column found.")

    if label_col is not None:
        plot_summary(df, label_col, amount_col)
        make_eda_plots(df, label_col, amount_col)
        model, metrics, importance_df = build_and_evaluate_model(df, label_col, amount_col)
        report_text = build_report_text(kpi_info, metrics, importance_df)
        save_business_report(report_text, "fraud_summary.txt")
        save_summary_pdf(report_text, "summary_report.pdf")
        print(f"\nSaved business report and summary PDF.")
        print(f"The trained model is saved as {model_path}.")
    else:
        print("\nSkipping plots and model because no fraud label column is available.")


if __name__ == "__main__":
    main()