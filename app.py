import os
from flask import Flask, render_template, request
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import io, base64

app = Flask(__name__)

# Load ML model
model = joblib.load("fraud_model.pkl")
top_features = joblib.load("top_6_features.pkl")
encoders = joblib.load("fraud_encoders.pkl")
 
# Sample dataset for Return Ratio & Dashboard
df = pd.read_csv("ecommerce_return_refund_fraud_dataset_100k.csv")  # optional, only for charts

# Home (Welcome Page)
@app.route("/")
def welcome():
    return render_template("welcome.html")

# Main Dashboard Landing
@app.route("/home")
def home():
    return render_template("home.html")

# About
@app.route("/about")
def about():
    return render_template("about.html")

# Risk Prediction
@app.route("/risk-prediction", methods=["GET", "POST"])
def risk_prediction():
    prediction = None
    confidence = None

    error_msg = None

    if request.method == "POST":
        user_input = {}
        try:
            for feature in top_features:
                value = request.form.get(feature)
                if not value:
                    raise ValueError(f"Missing value for {feature}")
                
                # Try to cast to float, but keep as string if it's categorical
                try:
                    user_input[feature] = float(value)
                except ValueError:
                    user_input[feature] = value # Keep as string (e.g., 'Yes', 'High')

            input_df = pd.DataFrame([user_input])

            # Encode categorical features
            for col in input_df.columns:
                if col in encoders:
                    if input_df[col].iloc[0] in encoders[col].classes_:
                        input_df[col] = encoders[col].transform(input_df[col])
                    else:
                        input_df[col] = 0

            pred = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0][pred] * 100

            prediction = "⚠️ FRAUD DETECTED" if pred == 1 else "✅ NOT FRAUD"
            confidence = f"{prob:.2f}%"
        except Exception as e:
            error_msg = str(e)
            print(f"Error in risk_prediction: {e}")

    return render_template("risk_prediction.html",
                           features=top_features,
                           prediction=prediction,
                           confidence=confidence,
                           error=error_msg)

# Return Ratio Chart
@app.route("/return-ratio")
def return_ratio():
    import numpy as np
    
    # --- 1. Top Key Stats ---
    means = df.groupby("fraud_label")["return_ratio"].mean()
    legit_mean_rr = round(means.get(0, 0) * 100, 0) # e.g. 18
    fraud_mean_rr = round(means.get(1, 0) * 100, 0) # e.g. 72

    # Simulated Financials (assuming some avg order value logic)
    # Scale simulation: Total Refund ~ 4.2 Cr, Fraud Loss ~ 1.8 Cr
    total_refund_amount = "4.2Cr" 
    fraud_refund_loss = "1.8Cr"

    # --- 2. Return Reasons Distribution (Simulated) ---
    reasons_labels = ["Not as described", "Fake item", "Wrong item", "Damaged", "No longer needed"]
    reasons_data = [28, 22, 18, 17, 15] # percentages
    reasons_colors = ['#0969da', '#cf222e', '#9a6700', '#1a7f37', '#8c959f']

    # --- 3. Return Rate by Category (Grouped Bar) ---
    # Real data calculation for category rates
    cat_means = df.groupby(["product_category", "fraud_label"])["return_ratio"].mean().unstack()
    # Normalize to percentage and handle missing
    cat_means = cat_means.fillna(0) * 100
    
    # Filter top 5 categories for cleaner chart
    top_cats = cat_means.sum(axis=1).nlargest(5).index.tolist()
    cat_labels = top_cats
    cat_fraud_data = [round(cat_means.loc[c, 1], 1) for c in top_cats]
    cat_legit_data = [round(cat_means.loc[c, 0], 1) for c in top_cats]

    # --- 4. Weekly Return Trends (Simulated) ---
    weeks = ['W1', 'W2', 'W3', 'W4', 'W5', 'W6']
    returns_count = [250, 310, 280, 350, 290, 400]
    refund_amount = [320, 430, 380, 510, 390, 580] # In '00s or similar scale for dual axis

    return render_template("return_ratio.html", 
                           legit_mean_rr=int(legit_mean_rr), 
                           fraud_mean_rr=int(fraud_mean_rr),
                           total_refund_amount=total_refund_amount,
                           fraud_refund_loss=fraud_refund_loss,
                           reasons_labels=reasons_labels,
                           reasons_data=reasons_data,
                           reasons_colors=reasons_colors,
                           cat_labels=cat_labels,
                           cat_fraud_data=cat_fraud_data,
                           cat_legit_data=cat_legit_data,
                           weeks=weeks,
                           returns_count=returns_count,
                           refund_amount=refund_amount)

# Batch Prediction (CSV Upload)
@app.route("/batch-prediction", methods=["GET", "POST"])
def batch_prediction():
    predictions = None
    if request.method == "POST":
        if 'file' not in request.files:
            return render_template("batch_prediction.html", error="No file part")
        file = request.files['file']
        if file.filename == '':
            return render_template("batch_prediction.html", error="No selected file")
        
        if file:
            try:
                # Read CSV
                batch_df = pd.read_csv(file)
                
                # Check for required columns (simple validation)
                missing_cols = [col for col in top_features if col not in batch_df.columns]
                if missing_cols:
                   return render_template("batch_prediction.html", error=f"Missing columns: {', '.join(missing_cols)}")

                # Process and Predict
                results = []
                for index, row in batch_df.iterrows():
                    input_data = row[top_features].to_dict()
                    input_df = pd.DataFrame([input_data])
                    
                    # Encode
                    for col in input_df.columns:
                        if col in encoders:
                            if input_df[col].iloc[0] in encoders[col].classes_:
                                input_df[col] = encoders[col].transform(input_df[col])
                            else:
                                input_df[col] = 0
                    
                    pred = model.predict(input_df)[0]
                    prob = model.predict_proba(input_df)[0][pred] * 100
                    
                    results.append({
                        "id": index + 1, # or transaction_id if available
                        "data": row.to_dict(),
                        "prediction": "FRAUD" if pred == 1 else "SAFE",
                        "confidence": f"{prob:.2f}%",
                        "is_fraud": pred == 1
                    })
                
                predictions = results
                
            except Exception as e:
                return render_template("batch_prediction.html", error=f"Error processing file: {str(e)}")

    return render_template("batch_prediction.html", predictions=predictions)

# Dashboard (example metrics)
@app.route("/dashboard")
def dashboard():
    import numpy as np
    
    # Check for date range filter
    time_range = request.args.get('range', '12m') # default to 12 months

    # 1. Calculate Stats based on time range
    if time_range == '30d':
        sample_fraction = 30 / 365
        sampled_df = df.sample(frac=sample_fraction, random_state=42)
        trend_title = "Last 30 Days Trend"
        periods = 30
        freq = 'D'
        date_format = '%d %b'
    else:
        sampled_df = df
        trend_title = "Monthly Transaction Trends"
        periods = 12
        freq = 'M'
        date_format = '%b'
    
    total_orders = len(sampled_df)
    fraud_count = sampled_df['fraud_label'].sum()
    legit_count = total_orders - fraud_count
    detection_rate = round((fraud_count / total_orders) * 100, 2) if total_orders > 0 else 0

    # 2. Synthesize Trends
    dates = pd.date_range(end=pd.Timestamp.now(), periods=periods, freq=freq).strftime(date_format)
    labels = dates.tolist()
    
    if time_range == '30d':
        legit_trend = [int(legit_count / 30 * (1 + 0.3 * np.random.randn())) for _ in range(30)]
        fraud_trend = [int(fraud_count / 30 * (1 + 0.3 * np.random.randn())) for _ in range(30)]
    else:
        legit_trend = [int(legit_count / 12 * (1 + 0.2 * np.random.randn())) for _ in range(12)]
        fraud_trend = [int(fraud_count / 12 * (1 + 0.2 * np.random.randn())) for _ in range(12)]

    # 3. New Charts Data: Fraud by Payment Method
    payment_counts = sampled_df[sampled_df['fraud_label'] == 1]['payment_method'].value_counts()
    payment_labels = payment_counts.index.tolist()
    payment_data = payment_counts.values.tolist()

    # 4. New Charts Data: Fraud by Category
    category_counts = sampled_df[sampled_df['fraud_label'] == 1]['product_category'].value_counts().head(5)
    category_labels = category_counts.index.tolist()
    category_data = category_counts.values.tolist()

    # 5. New Charts Data: Fraud by Device Type
    device_counts = sampled_df[sampled_df['fraud_label'] == 1]['device_type'].value_counts()
    device_labels = device_counts.index.tolist()
    device_data = device_counts.values.tolist()

    # 6. New Charts Data: Return Reasons (Overall)
    reason_counts = sampled_df['return_reason'].value_counts().head(5)
    reason_labels = reason_counts.index.tolist()
    reason_data = reason_counts.values.tolist()

    # 7. Recent Transactions with Simulated Person Details
    import random
    names = ["James Smith", "Maria Rodriguez", "David Kim", "Sarah Johnson", "Michael Chen", "Emily Davis", "Robert Wilson", "Jennifer Lee"]
    domains = ["gmail.com", "yahoo.com", "outlook.com", "icloud.com"]
    cities = ["New York, NY", "Los Angeles, CA", "Chicago, IL", "Houston, TX", "Phoenix, AZ"]

    recent_transactions = sampled_df[['order_id', 'customer_id', 'product_category', 'order_value', 'payment_method', 'fraud_label', 'return_ratio', 'device_type', 'customer_account_age_days']].head(10).to_dict(orient='records')
    
    for tx in recent_transactions:
        # Simulate Person Details
        first_name = random.choice(names).split()[0]
        last_name = random.choice(names).split()[1]
        tx['name'] = f"{first_name} {last_name}"
        tx['email'] = f"{first_name.lower()}.{last_name.lower()}@{random.choice(domains)}"
        tx['location'] = random.choice(cities)
        tx['age'] = random.randint(22, 55)
        tx['gender'] = random.choice(["Male", "Female"])
        tx['phone'] = f"+1 ({random.randint(200, 999)}) {random.randint(100, 999)}-{random.randint(1000, 9999)}"

    return render_template("dashboard.html",
                           total_orders=f"{total_orders:,}",
                           fraud_count=f"{fraud_count:,}",
                           legit_count=f"{legit_count:,}",
                           detection_rate=detection_rate,
                           labels=labels,
                           legit_trend=legit_trend,
                           fraud_trend=fraud_trend,
                           trend_title=trend_title,
                           current_range=time_range,
                           payment_labels=payment_labels,
                           payment_data=payment_data,
                           category_labels=category_labels,
                           category_data=category_data,
                           device_labels=device_labels,  # New
                           device_data=device_data,      # New
                           reason_labels=reason_labels,  # New
                           reason_data=reason_data,      # New
                           recent_transactions=recent_transactions)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
