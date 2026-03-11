import joblib
import pandas as pd

# Load ML model and encoders
model = joblib.load("fraud_model.pkl")
top_features = joblib.load("top_6_features.pkl")
encoders = joblib.load("fraud_encoders.pkl")

# Test Scenarios
scenarios = [
    {
        "name": "Scenario 1: User's reported High Risk",
        "data": {
            "alert_generated": 1,
            "return_ratio": 0.8,
            "alert_severity": 2,
            "total_returns": 15,
            "total_orders": 20,
            "refund_amount": 5000
        }
    },
    {
        "name": "Scenario 2: Low Risk (Clean History)",
        "data": {
            "alert_generated": 0,
            "return_ratio": 0.05,
            "alert_severity": 0,
            "total_returns": 0,
            "total_orders": 100,
            "refund_amount": 50
        }
    },
    {
        "name": "Scenario 4: Extreme High Risk",
        "data": {
            "alert_generated": 1,
            "return_ratio": 0.95,
            "alert_severity": 2,
            "total_returns": 80,
            "total_orders": 85,
            "refund_amount": 12000
        }
    }
]

results_log = []
def log_result(name, data, res, conf, debug_info):
    results_log.append(f"Scenario: {name}\nResult: {res}\nConfidence: {conf}\nInputs: {data}\nDebug: {debug_info}\n\n")

print("--- MODEL PREDICTION TESTS ---")
for s in scenarios:
    # Ensure column order matches top_features
    input_df = pd.DataFrame([s['data']])[top_features]
    
    # Encode categorical features
    for col in input_df.columns:
        if col in encoders:
            if input_df[col].iloc[0] in encoders[col].classes_:
                input_df[col] = encoders[col].transform(input_df[col])
            else:
                input_df[col] = 0
    
    pred = model.predict(input_df)[0]
    probs = model.predict_proba(input_df)[0]
    prob = probs[pred] * 100
    res = "FRAUD" if pred == 1 else "SAFE"
    debug_info = f"Classes={model.classes_}, Probs={probs}, Pred={pred}"
    
    log_result(s['name'], s['data'], res, f"{prob:.2f}%", debug_info)
    print(f"{s['name']}: {res}")

with open("test_results_debug.txt", "w") as f:
    f.writelines(results_log)
print("Results saved to test_results_debug.txt")
