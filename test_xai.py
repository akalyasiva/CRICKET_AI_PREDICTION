import time
import numpy as np

from utils.prediction_engine import load_models, preprocess
from utils.shap_engine import create_shap_explainer, local_shap_values
from utils.lime_engine import create_lime_explainer, local_lime_explanation

# ---------------- LOAD MODELS ----------------
lstm_model, bilstm_model, scaler, enc = load_models()

# ---------------- SAMPLE INPUT ----------------
inning = 2
batting = "Chennai Super Kings"
bowling = "Mumbai Indians"
overs = 12.0
score = 110
wickets = 3

ball_number = int(overs * 6)
run_rate = score / overs
remaining = 20 - overs

# Preprocess
X_flat = preprocess(inning, batting, bowling, ball_number,
                    score, wickets, run_rate, remaining,
                    scaler, enc)

# Feature names (IMPORTANT for readability)
feature_names = [
    "Inning", "Batting Team", "Bowling Team", "Ball Number",
    "Current Score", "Wickets", "Run Rate", "Remaining Overs"
]

# ---------------- SHAP GLOBAL ----------------
print("\n===== SHAP GLOBAL EXPLANATION =====")
start = time.time()

background = np.random.randn(50, 8)
shap_explainer = create_shap_explainer(lstm_model, background)

# Generate sample dataset for global explanation
sample_data = np.random.randn(20, 8)

# Compute SHAP values for multiple samples
shap_values_global = shap_explainer.shap_values(sample_data)

# Convert to absolute mean importance
mean_importance = np.mean(np.abs(shap_values_global), axis=0)

end = time.time()

print("Method Used: KernelExplainer (Shapley Values)")
print(f"Time Taken: {end - start:.4f} seconds")
print("Global Feature Importance:")

for f, v in zip(feature_names, mean_importance):
    print(f"{f}: {v:.4f}")

# ---------------- SHAP LOCAL ----------------
print("\n===== SHAP LOCAL EXPLANATION =====")
start = time.time()

shap_result = local_shap_values(shap_explainer, X_flat)

end = time.time()

print(f"Time Taken: {end - start:.4f} seconds")
print("Feature Contributions (Local):")

for f, v in zip(shap_result["feature_names"], shap_result["shap_values"]):
    print(f"{f}: {v:.4f}")

# ---------------- LIME LOCAL ----------------
print("\n===== LIME LOCAL EXPLANATION =====")
start = time.time()

lime_explainer = create_lime_explainer(background)
lime_result = local_lime_explanation(lime_explainer, lstm_model, X_flat)

end = time.time()

print("Method Used: LimeTabularExplainer (Perturbation + Linear Model)")
print(f"Time Taken: {end - start:.4f} seconds")
print("Feature Weights (Local):")

for k, v in lime_result["weights"].items():
    print(f"{k}: {v:.4f}")

# ---------------- FINAL COMPARISON ----------------
print("\n===== XAI COMPARISON =====")
print("SHAP: Global + Local explanation, high consistency, slower")
print("LIME: Local explanation only, faster but slightly unstable")