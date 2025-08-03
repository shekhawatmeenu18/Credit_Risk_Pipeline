import os
import matplotlib.pyplot as plt
import seaborn as sns

def eda_plots(df, target='default', output_dir="reports/eda"):
    os.makedirs(output_dir, exist_ok=True)

    # Separate numeric and categorical columns (excluding target)
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop(target)
    categorical_cols = df.select_dtypes(include=['object']).columns

    print(f"Numeric Columns: {list(numeric_cols)}")
    print(f"Categorical Columns: {list(categorical_cols)}")

    # Plot numeric columns
    for col in numeric_cols:
        plt.figure(figsize=(8, 5))
        sns.histplot(df[col], kde=True, bins=30, color="skyblue")
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{col}_hist.png"))
        plt.close()

    # Plot categorical columns
    for col in categorical_cols:
        plt.figure(figsize=(8, 5))
        sns.countplot(x=df[col], order=df[col].value_counts().index, palette="viridis")
        plt.title(f"Distribution of {col}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{col}_bar.png"))
        plt.close()

    # Correlation heatmap for numeric columns
    plt.figure(figsize=(10, 6))
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
    plt.close()

    print(f"EDA plots saved in: {output_dir}")