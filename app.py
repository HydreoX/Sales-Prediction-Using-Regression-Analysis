from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("models/sales_prediction_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Get values from form
    data = {
        "Item_Weight": float(request.form["Item_Weight"]),
        "Item_Fat_Content": request.form["Item_Fat_Content"],
        "Item_Visibility": float(request.form["Item_Visibility"]),
        "Item_Type": request.form["Item_Type"],
        "Item_MRP": float(request.form["Item_MRP"]),
        "Outlet_Identifier": request.form["Outlet_Identifier"],
        "Outlet_Establishment_Year": int(
            request.form["Outlet_Establishment_Year"]
        ),
        "Outlet_Size": request.form["Outlet_Size"],
        "Outlet_Location_Type": request.form["Outlet_Location_Type"],
        "Outlet_Type": request.form["Outlet_Type"]
    }

    # Convert input into DataFrame
    input_data = pd.DataFrame([data])

    # Make prediction
    prediction = model.predict(input_data)[0]

    prediction = round(prediction, 2)

    # Simple business insight
    if prediction >= 2500:
        insight = "High expected sales"
    elif prediction >= 1500:
        insight = "Moderate expected sales"
    else:
        insight = "Low expected sales"

    return render_template(
        "result.html",
        prediction=prediction,
        insight=insight
    )

@app.route("/analytics")
def analytics():
    return render_template("analytics.html")

if __name__ == "__main__":
    app.run(debug=True)