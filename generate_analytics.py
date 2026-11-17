import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


# ==========================================
# 1. LOAD DATA
# ==========================================

data = pd.read_csv("Dataset/train.csv")

print("Dataset loaded.")


# ==========================================
# 2. SAME CLEANING USED DURING TRAINING
# ==========================================

data["Item_Fat_Content"] = data["Item_Fat_Content"].replace(
    {
        "LF": "Low Fat",
        "low fat": "Low Fat",
        "reg": "Regular"
    }
)

data["Item_Visibility"] = data["Item_Visibility"].replace(
    0, np.nan
)


# ==========================================
# 3. FEATURES AND TARGET
# ==========================================

X = data.drop("Item_Outlet_Sales", axis=1)
y = data["Item_Outlet_Sales"]

X = X.drop("Item_Identifier", axis=1)


# ==========================================
# 4. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ==========================================
# 5. LOAD BEST MODEL
# ==========================================

model = joblib.load(
    "models/sales_prediction_model.pkl"
)

print("Gradient Boosting model loaded.")


# ==========================================
# 6. PREDICTIONS
# ==========================================

predictions = model.predict(X_test)


# ==========================================
# 7. MODEL COMPARISON GRAPH
# ==========================================

comparison = pd.read_csv("model_comparison.csv")

plt.figure(figsize=(10, 6))

plt.bar(
    comparison["Model"],
    comparison["R2"]
)

plt.title("Regression Model Comparison")
plt.xlabel("Model")
plt.ylabel("R² Score")

plt.xticks(
    rotation=20,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    "static/images/model_comparison.png",
    dpi=150
)

plt.close()


# ==========================================
# 8. ACTUAL VS PREDICTED GRAPH
# ==========================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    predictions,
    alpha=0.5
)

# Perfect prediction line
minimum = min(
    y_test.min(),
    predictions.min()
)

maximum = max(
    y_test.max(),
    predictions.max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.title("Actual vs Predicted Sales")

plt.xlabel("Actual Sales")

plt.ylabel("Predicted Sales")

plt.tight_layout()

plt.savefig(
    "static/images/actual_vs_predicted.png",
    dpi=150
)

plt.close()


# ==========================================
# 9. FEATURE IMPORTANCE
# ==========================================

pipeline = model

preprocessor = pipeline.named_steps["preprocessor"]

regressor = pipeline.named_steps["model"]

feature_names = (
    preprocessor
    .get_feature_names_out()
)

importance = regressor.feature_importances_

feature_df = pd.DataFrame(
    {
        "Feature": feature_names,
        "Importance": importance
    }
)

feature_df = feature_df.sort_values(
    by="Importance",
    ascending=False
)

# Top 10 features
top_features = feature_df.head(10)

plt.figure(figsize=(10, 6))

plt.barh(
    top_features["Feature"][::-1],
    top_features["Importance"][::-1]
)

plt.title(
    "Top 10 Features Influencing Sales Prediction"
)

plt.xlabel("Feature Importance")

plt.tight_layout()

plt.savefig(
    "static/images/feature_importance.png",
    dpi=150
)

plt.close()


# ==========================================
# 10. SAVE FEATURE IMPORTANCE DATA
# ==========================================

feature_df.to_csv(
    "feature_importance.csv",
    index=False
)

print("\nAnalytics generated successfully!")

print(
    "Created:"
)

print("1. model_comparison.png")
print("2. actual_vs_predicted.png")
print("3. feature_importance.png")