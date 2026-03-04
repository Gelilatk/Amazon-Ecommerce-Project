
#########################  analysis tools  ##########################
#importing essential libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
# -------------------------------------------------
# 1) Load Dataset
# -------------------------------------------------
def load_data(filepath):
    """Load dataset from CSV file."""
    return pd.read_csv(filepath)
    


# -------------------------------------------------
# 2) DATA QUALITY CHECK
# -------------------------------------------------
def audit_data(df):
    """
    Quick dataset health check.
    Returns dictionary for programmatic use
    AND prints a readable summary for analysis.
    """

    summary = {
        "shape": df.shape,
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum())
    }

    # ---------- Report ----------
    print("DATA AUDIT REPORT")
    print("-" * 50)

    # Print the Shape of the dataframe
    print(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

    # print the Data types and count of column
    print("\nColumn Types:")
    print(df.dtypes.value_counts())

    # Missing values (only show columns that have missing or filter the column with missing value)
    missing = df.isnull().sum()
    missing = missing[missing > 0]

    print("\nMissing Values:")
    if len(missing) == 0:
        print("No missing values")
    else:
        print(missing.sort_values(ascending=False))

    # Duplicates
    print(f"\nDuplicate Rows: {summary['duplicate_rows']}")

    print("-" * 50)

    return summary

# -------------------------------------------------
# 3) GENERIC NUMBER CLEANER
# -------------------------------------------------
def clean_numeric_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Removes non-numeric characters (currency signs, commas, %, text)
    and converts columns to numeric safely.
    Works for ANY currency or percent format.
    """
    for col in columns:
        df[col] = (
            df[col]
            .astype(str) # check that every value in the column is treated as text
            .str.replace(r"[^\d.]", "", regex=True)  # keep only digits and decimal point
        )
        df[col] = pd.to_numeric(df[col], errors="coerce") #converts the cleaned strings into proper numeric values
    return df

# -------------------------------------------------
# 5) Cleanig Text
# -------------------------------------------------

# Function to standardize review text for analysis
def clean_review_text(text):

    # Keep missing values unchanged
    if pd.isna(text):
        return text

    # Convert text to lowercase so words are counted consistently
    text = text.lower()

   # Remove URLs (links in reviews like amazon shortened links)
    text = re.sub(r'http\S+|www\S+', ' ', text)

   # Replace punctuation with space so words don’t merge together
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

   # Normalize multiple spaces and trim leading/trailing spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# -------------------------------------------------
# 6) Reusable Module
# -------------------------------------------------

def plot_scatter_with_correlation(df, x_col, y_col, title, x_label, y_label):
#Reusable scatter plot with correlation calculation.
 plt.figure(figsize=(8,5))
 sns.scatterplot(data=df, x=x_col, y=y_col, alpha=0.6)
 plt.title(title)
 plt.xlabel(x_label)
 plt.ylabel(y_label)
 plt.tight_layout()
 plt.show()
 correlation = df[x_col].corr(df[y_col])
 print(f"Correlation: {correlation:.2f}")


def plot_histogram(df, column, title, x_lable, y_lable,bins=30, kde=True):
# Reusable histPlot
 sns.histplot(df[column], bins=bins)
 plt.title(title)
 plt.xlabel(x_lable)
 plt.ylabel(y_lable)
 plt.show()


# Reusable bar plot
def plot_bar(
    df,
    group_col,              # column to group by (e.g., 'category' or 'product_name')
    count_col=None,         # column to count (e.g., 'review_content'); if None, use value_counts
    n=10,
    mode='top',             # 'top' or 'bottom'
    title=None,
    xlabel="Count",
    ylabel=None,
    figsize=(6,4)
):
    if count_col:
        counts = (
            df.groupby(group_col)[count_col]
              .count()
              .sort_values(ascending=False)
        )
    else:
        counts = df[group_col].value_counts()

    if mode == 'top':
        data = counts.head(n)
        default_title = f"Top {n} {group_col.title()}s"
    else:
        data = counts.tail(n)
        default_title = f"Bottom {n} {group_col.title()}s"

    plt.figure(figsize=figsize)
    sns.barplot(x=data.values, y=data.index)
    plt.title(title if title else default_title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel if ylabel else group_col.title())
    plt.show()


# Reusable heatmap
def plot_correlation_heatmap(df, columns, title="Correlation Matrix", cmap="coolwarm"):
    corr_matrix = df[columns].corr()
    plt.figure(figsize=(6,4))
    sns.heatmap(corr_matrix, annot=True, cmap=cmap, fmt=".2f")
    plt.title(title)
    plt.show()