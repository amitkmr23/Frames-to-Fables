import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)

    # Encode categorical features
    le = LabelEncoder()
    df['card'] = le.fit_transform(df['card'])         # yes/no -> 1/0
    df['owner'] = le.fit_transform(df['owner'])
    df['selfemp'] = le.fit_transform(df['selfemp'])

    # Split features and labels
    X = df.drop('card', axis=1)
    y = df['card']

    # Train-test split
    return train_test_split(X, y, test_size=0.2, random_state=42)
