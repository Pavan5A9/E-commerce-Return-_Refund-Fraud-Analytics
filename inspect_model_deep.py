import joblib
import pandas as pd
import numpy as np

model = joblib.load("fraud_model.pkl")
top_features = joblib.load("top_6_features.pkl")
encoders = joblib.load("fraud_encoders.pkl")

with open("model_inspection_results.txt", "w") as f:
    f.write(f"Model Classes: {model.classes_}\n")
    f.write(f"Top Features: {top_features}\n")

    for feat in encoders:
        f.write(f"Encoder for {feat} classes: {encoders[feat].classes_}\n")

    data = {
        'alert_generated': 1,
        'return_ratio': 0.1,
        'alert_severity': 2,
        'total_returns': 10,
        'total_orders': 15,
        'refund_amount': 5000
    }

    input_df = pd.DataFrame([data])[top_features]
    f.write("\nRaw Input DF Types:\n")
    f.write(f"{input_df.dtypes}\n")

    # Try with and without encoding
    f.write("\n--- TEST: WITHOUT ENCODING ---\n")
    pred = model.predict(input_df)[0]
    probs = model.predict_proba(input_df)[0]
    f.write(f"Pred: {pred}, Probs: {probs}\n")

    input_df_enc = input_df.copy()
    for col in input_df_enc.columns:
        if col in encoders:
            val = input_df_enc[col].iloc[0]
            # Try to match numeric input to string classes if needed
            if str(val) in [str(c) for c in encoders[col].classes_]:
                # Find the index of the matching class string
                class_list = [str(c) for c in encoders[col].classes_]
                idx = class_list.index(str(val))
                actual_val = encoders[col].classes_[idx]
                input_df_enc[col] = encoders[col].transform([actual_val])
            else:
                f.write(f"VAL {val} NOT IN {col} CLASSES!\n")

    f.write("\n--- TEST: WITH ENCODING ---\n")
    pred_enc = model.predict(input_df_enc)[0]
    probs_enc = model.predict_proba(input_df_enc)[0]
    f.write(f"Pred: {pred_enc}, Probs: {probs_enc}\n")
    f.write(f"Encoded Input DF Values:\n{input_df_enc.iloc[0]}\n")

print("Done. Results in model_inspection_results.txt")
