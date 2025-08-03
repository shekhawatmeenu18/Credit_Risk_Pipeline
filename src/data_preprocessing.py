import pandas as pd
import re
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# ----------------------------
# Data Loading & Preprocessing
# ----------------------------
def load_data(path):
    """Load dataset from CSV and perform feature engineering"""
    df = pd.read_csv(path, index_col=0)

    # Standardize credit_history categories
    df['credit_history'] = df['credit_history'].replace({
        'fully repaid this bank': 'repaid',
        'fully repaid': 'repaid'
    })

    # Employment feature engineering
    df['employment_months'] = df['employment_length'].apply(convert_to_months)
    df['employment_bucket'] = df['employment_months'].apply(bucket_employment)

    # Residence feature engineering
    df['residence_months'] = df['residence_history'].apply(convert_residence_to_months)
    df['residence_bucket'] = df['residence_months'].apply(bucket_residence)

    # Drop unnecessary columns
    df.drop(columns=['telephone', 'employment_length', 'residence_history'], inplace=True)

    # Show summary
    print("\n✅ Data Loaded and Processed:")
    print(f"Shape: {df.shape}")
    print("Object Columns:", df.select_dtypes(include=['object']).columns.tolist())
    print("Numeric Columns:", df.select_dtypes(exclude=['object']).columns.tolist())

    return df


# ----------------------------
# Conversion Functions
# ----------------------------
def convert_to_months(x):
    """Convert 'x years' or 'x months' to total months"""
    if pd.isnull(x):
        return None
    x = str(x).lower().strip()
    if 'month' in x:
        num = int(re.search(r'\d+', x).group())
        return num
    elif 'year' in x:
        num = int(re.search(r'\d+', x).group())
        return num * 12
    return None


def bucket_employment(months):
    """Create employment duration buckets"""
    if months is None:
        return 'Unknown'
    if months < 12:
        return '<1 year'
    elif 12 <= months < 36:
        return '1-3 years'
    elif 36 <= months < 84:
        return '3-7 years'
    else:
        return '7+ years'


def convert_residence_to_months(x):
    """Convert residence duration to months"""
    if pd.isnull(x):
        return None
    x = str(x).lower().strip()
    if 'month' in x:
        num = int(re.search(r'\d+', x).group())
        return num
    elif 'year' in x:
        num = int(re.search(r'\d+', x).group())
        return num * 12
    return None


def bucket_residence(months):
    """Create residence duration buckets"""
    if months is None:
        return 'Unknown'
    if months < 6:
        return '<6 months'
    elif 6 <= months < 12:
        return '6-12 months'
    elif 12 <= months < 36:
        return '1-3 years'
    elif 36 <= months < 84:
        return '3-7 years'
    else:
        return '7+ years'


# ----------------------------
# Auto Feature Detection
# ----------------------------
def get_feature_lists(df, target='default'):
    """Automatically detect numeric and categorical features (excluding target)"""
    numeric_features = df.select_dtypes(include=['number']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()

    # Remove target column if present
    if target in numeric_features:
        numeric_features.remove(target)
    if target in categorical_features:
        categorical_features.remove(target)

    return numeric_features, categorical_features


# ----------------------------
# Preprocessor
# ----------------------------
def create_preprocessor(numeric_features, categorical_features):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    return ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )


# ----------------------------
# Train-Test Split
# ----------------------------
def split_data(df, target='default', test_size=0.2, random_state=42):
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
