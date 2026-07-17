import joblib
import numpy as np
from app.questionnaire import get_terminal_input
from app.utils import risk_level

def run_cli():
    # 1. Load your trained model
    try:
        model = joblib.load("models/best_model.pkl")
    except:
        print("Error: Model file not found. Run model.py first!")
        return

    # 2. Get answers from terminal
    input_data = get_terminal_input()

    # 3. Predict probability
    prob = model.predict_proba(input_data)[0][1]

    # 4. Show Result
    risk = risk_level(prob)
    print("\n" + "="*50)
    print(f"RESULT: {risk}")
    print(f"Confidence Score: {prob:.2f}")
    print("="*50)

if __name__ == "__main__":
    run_cli()