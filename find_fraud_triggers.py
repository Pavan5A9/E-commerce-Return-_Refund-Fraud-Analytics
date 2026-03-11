import joblib
import pandas as pd
import numpy as np

model = joblib.load("fraud_model.pkl")
top_features = joblib.load("top_6_features.pkl")

def check_fraud(data):
    input_df = pd.DataFrame([data])[top_features]
    pred = model.predict(input_df)[0]
    return pred == 1

results = []

# Try different combinations
for rr in [0.1, 0.5, 0.9]:
    for ra in [100, 5000, 50000]:
        for tr in [0, 10, 50]:
            data = {
                "alert_generated": 1,
                "return_ratio": rr,
                "alert_severity": 2,
                "total_returns": tr,
                "total_orders": tr + 5,
                "refund_amount": ra
            }
            if check_fraud(data):
                results.append(data)

with open("fraud_triggers.txt", "w") as f:
    if not results:
        f.write("No fraud cases found in simple grid search.\n")
        # Try a more random/extreme search
        for i in range(100):
            data = {
                "alert_generated": np.random.randint(0, 2),
                "return_ratio": np.random.random(),
                "alert_severity": np.random.randint(0, 3),
                "total_returns": np.random.randint(0, 500),
                "total_orders": np.random.randint(1, 1000),
                "refund_amount": np.random.randint(0, 100000)
            }
            if check_fraud(data):
                f.write(f"FOUND FRAUD: {data}\n")
    else:
        for r in results:
            f.write(f"FRAUD CASE: {r}\n")

print("Done searching.")
