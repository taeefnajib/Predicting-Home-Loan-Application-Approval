# Importing all dependencies
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pickle
from dataclasses_json import dataclass_json
from dataclasses import dataclass

@dataclass_json
@dataclass
class Hyperparameters(object):
    data_filepath: str = "data/raw.csv"
    model_filepath: str = "model/model.pkl"
    test_size: float = 0.2
    random_state: int = 6
    max_depth: int = 4
    min_samples_leaf: int = 2 
    min_samples_split: int = 2 
    n_estimators: int = 127

hp = Hyperparameters()

# Creating the dataframe
def create_df(data_filepath):
    return pd.read_csv(data_filepath)

# Cleaning dataset
def clean_ds(df):
    # Cleaning dataset
    df.drop(columns=["Loan_ID"], axis=1, inplace=True)
    df.dropna(subset=["Gender","Married",], axis=0, inplace=True)
    df.fillna(value={"Dependents":"Unknown", "Self_Employed":"Unknown",}, inplace=True)
    df["LoanAmount"].fillna(df["LoanAmount"].mean(), inplace=True)
    df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mean(), inplace=True)
    df["Credit_History"].fillna(df["Credit_History"].mean(), inplace=True)
    df["Loan_Status"].replace({"Y":1, "N":0}, inplace=True)
    return df

# Handling categorical columns
def handle_cat_cols(df):
    cat_cols = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
    df = pd.get_dummies(data=df, columns=cat_cols)
    return df

# Pre-processing dataset
def preprocess_ds(df):
    df = clean_ds(df)
    df = handle_cat_cols(df)
    df.to_csv("data/processed.csv", index=False)
    return df

# Splitting dataset into train, validation and test data
def split_train_test(df, test_size, random_state):
    X = df.drop(["Loan_Status"], axis = 1)
    y = df["Loan_Status"]
    return train_test_split(X, y, test_size = test_size, random_state = random_state)


# Training model on train data
def train_model(X_train, y_train,
                n_estimators,
                min_samples_split,
                min_samples_leaf,
                max_depth,
                random_state,
                model_filepath):
    clf = RandomForestClassifier(max_depth=max_depth,
                                min_samples_leaf=min_samples_leaf, 
                                min_samples_split=min_samples_split, 
                                n_estimators=n_estimators,
                                random_state=random_state)
    clf.fit(X_train, y_train)
    pickle.dump(clf, open(model_filepath, "wb"))
    return clf

# Predicting on test data
def predict(X_test, y_test, model_filepath):
    model = pickle.load(open(model_filepath, "rb"))
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy Score for GradientBoostingClassifier: {acc}")
    return acc


def run_wf(hp: Hyperparameters) -> RandomForestClassifier:
    df = create_df(hp.data_filepath)
    df = preprocess_ds(df=df)
    X_train, X_test, y_train, y_test = split_train_test(df=df, test_size=hp.test_size, random_state=hp.random_state)
    return train_model(X_train=X_train, y_train=y_train,
                        n_estimators=hp.n_estimators,
                        min_samples_leaf=hp.min_samples_leaf,
                        min_samples_split=hp.min_samples_split,
                        max_depth=hp.max_depth,
                        random_state=hp.random_state,
                        model_filepath=hp.model_filepath)

# predict(X_test=X_test, y_test=y_test, model_filepath=hp.model_filepath)

if __name__=="__main__":
    run_wf(hp=hp)