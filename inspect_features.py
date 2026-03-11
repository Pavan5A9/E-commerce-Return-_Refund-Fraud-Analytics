import joblib
features = joblib.load('top_6_features.pkl')
print("--- START FEATURES ---")
for f in features:
    print(f)
print("--- END FEATURES ---")
