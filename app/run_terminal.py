import os
import joblib
from app.questionnaire import get_terminal_input, FEATURE_ORDER
from app.utils import risk_level

def main():
    model_path = "models/best_model.pkl"
    scaler_path = "models/scaler.pkl"

    # -------------------------
    # Load model
    # -------------------------
    if not os.path.exists(model_path):
        print("Model not found. Run train.py first.")
        return

    if not os.path.exists(scaler_path):
        print("Scaler not found. Train model again.")
        return

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    print("Model loaded successfully!")

    # -------------------------
    # Get input
    # -------------------------
    X = get_terminal_input()

    # -------------------------
    # SCALE INPUT  ✅ FIX
    # -------------------------
    X_scaled = scaler.transform(X)

    # -------------------------
    # ML prediction
    # -------------------------
    prob = model.predict_proba(X_scaled)[0][1]

    # -------------------------
    # HYBRID LOGIC
    # -------------------------
    aq_sum = sum(X[0][:10])

    delay = X[0][FEATURE_ORDER.index("social_communication_delay")]

    repetitive = X[0][FEATURE_ORDER.index("repetitive_behaviors_score")]

    # Strong HIGH override
    if delay >= 7 and aq_sum >= 4:
        prob = max(prob, 0.75)

    # Strong LOW override
    if aq_sum == 0 and delay == 0:
        prob = min(prob, 0.25)

    # Mild stabilization
    if 2 <= aq_sum <= 4 and delay < 5:
        prob = min(max(prob, 0.4), 0.65)

    # -------------------------
    # Risk label
    # -------------------------
    risk = risk_level(prob)

    # -------------------------
    # Output
    # -------------------------
    print("\n" + "=" * 50)
    print("           SCREENING RESULTS")
    print("=" * 50)
    print(f"Risk Level:        {risk}")
    print(f"Probability Score: {prob:.4f}")
    print("=" * 50)

if __name__ == "__main__":
    main()