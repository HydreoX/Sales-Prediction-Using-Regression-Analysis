import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# 1. LOAD DATASET
# ==========================================

data = pd.read_csv("Dataset/train.csv")

print("\nDataset loaded successfully!")
print("Dataset shape:", data.shape)


# ==========================================
# 2. DATA CLEANING
# ==========================================

# Standardize Item_Fat_Content values
data["Item_Fat_Content"] = data["Item_Fat_Content"].replace(
    {
        "LF": "Low Fat",
        "low fat": "Low Fat",
        "reg": "Regular"
    }
)

# Treat zero visibility as missing
data["Item_Visibility"] = data["Item_Visibility"].replace(
    0, np.nan
)


# ==========================================
# 3. FEATURES AND TARGET
# ==========================================

X = data.drop("Item_Outlet_Sales", axis=1)

y = data["Item_Outlet_Sales"]


# Remove identifier
X = X.drop("Item_Identifier", axis=1)


# ==========================================
# 4. IDENTIFY COLUMN TYPES
# ==========================================

categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    exclude=["object"]
).columns.tolist()


print("\nCategorical columns:")
print(categorical_columns)

print("\nNumerical columns:")
print(numerical_columns)


# ==========================================
# 5. PREPROCESSING
# ==========================================

numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numerical_pipeline,
            numerical_columns
        ),
        (
            "cat",
            categorical_pipeline,
            categorical_columns
        )
    ]
)


# ==========================================
# 6. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 7. DEFINE MODELS
# ==========================================

models = {

    "Linear Regression":
        LinearRegression(),

    "Decision Tree":
        DecisionTreeRegressor(
            random_state=42,
            max_depth=12
        ),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),

    "Gradient Boosting":
        GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )
}


# ==========================================
# 8. TRAIN AND EVALUATE MODELS
# ==========================================

results = []

trained_models = {}


for name, model in models.items():

    print("\n----------------------------------")
    print("Training:", name)
    print("----------------------------------")

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    predictions = pipeline.predict(X_test)

    # Metrics
    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    results.append(
        {
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        }
    )

    trained_models[name] = pipeline

    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R2   : {r2:.4f}")


# ==========================================
# 9. CREATE COMPARISON TABLE
# ==========================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="R2",
    ascending=False
)


print("\n\n==========================================")
print("MODEL COMPARISON")
print("==========================================")

print(
    results_df.to_string(
        index=False,
        formatters={
            "MAE": "{:.2f}".format,
            "RMSE": "{:.2f}".format,
            "R2": "{:.4f}".format
        }
    )
)


# ==========================================
# 10. SELECT BEST MODEL
# ==========================================

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[
    best_model_name
]


print("\n==========================================")
print("BEST MODEL")
print("==========================================")

print("Model:", best_model_name)

print(
    "R2 Score:",
    f"{results_df.iloc[0]['R2']:.4f}"
)


# ==========================================
# 11. SAVE BEST MODEL
# ==========================================

joblib.dump(
    best_model,
    "models/sales_prediction_model.pkl"
)


# ==========================================
# 12. SAVE RESULTS
# ==========================================

results_df.to_csv(
    "model_comparison.csv",
    index=False
)


print("\nBest model saved successfully!")

print(
    "Location: models/sales_prediction_model.pkl"
)

print(
    "\nComparison saved to: model_comparison.csv"
)