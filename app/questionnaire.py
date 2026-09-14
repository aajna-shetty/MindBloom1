import numpy as np


FEATURE_ORDER = [
    "A1_Score", "A2_Score", "A3_Score", "A4_Score", "A5_Score",
    "A6_Score", "A7_Score", "A8_Score", "A9_Score", "A10_Score",
    "age", "gender", "jaundice", "family_history_asd", "used_app_before",
    "preeclampsia", "birth_asphyxia", "preterm_birth", "low_birth_weight",
    "consanguineous_marriage", "social_communication_delay",
    "repetitive_behaviors_score", "sensory_sensitivities",
    "first_words_months", "two_word_phrases_months",
    "eye_contact_age_months", "social_smile_months",
    "intellectual_disability", "epilepsy", "adhd",
    "language_disorder", "motor_delay"
]



QUESTIONS = [
    {
        "id": "A1_Score",
        "question": "Does your child have difficulty sharing interests or excitement with others?"
    },
    {
        "id": "A2_Score",
        "question": "Does your child avoid eye contact?"
    },
    {
        "id": "A3_Score",
        "question": "Does your child struggle to understand emotions?"
    },
    {
        "id": "A4_Score",
        "question": "Does your child have difficulty maintaining conversations?"
    },
    {
        "id": "A5_Score",
        "question": "Does your child fail to respond when their name is called?"
    },
    {
        "id": "A6_Score",
        "question": "Has your child experienced delayed speech development?"
    },
    {
        "id": "A7_Score",
        "question": "Does your child repeat words or phrases frequently?"
    },
    {
        "id": "A8_Score",
        "question": "Does your child struggle to express needs verbally?"
    },
    {
        "id": "A9_Score",
        "question": "Does your child display repetitive movements?"
    },
    {
        "id": "A10_Score",
        "question": "Does your child become upset when routines change?"
    }
]


def build_feature_vector_from_answers(answers):

    vector = []

    for feature in FEATURE_ORDER:
        value = answers.get(feature, 0)

        try:
            value = float(value)
        except:
            value = 0

        vector.append(value)

    return vector


def yn(question):
    return 1 if input(question).strip().lower() == "y" else 0


def show_age_group(age):

    if age <= 4:
        print("\nDevelopment Stage: Toddler (2-4 years)")
    elif age <= 9:
        print("\nDevelopment Stage: Child (5-9 years)")
    else:
        print("\nDevelopment Stage: Older Child (10-12 years)")


def get_terminal_input():

    print("\n" + "=" * 60)
    print("      MINDBLOOM AUTISM RISK SCREENING")
    print("=" * 60)

    answers = {}

    age = float(input("Child age (years): "))
    answers["age"] = age

    show_age_group(age)

    print("\nPlease answer with y or n.\n")

    aq = [
        yn("Difficulty sharing interests or excitement with others? "),
        yn("Avoids eye contact? "),
        yn("Struggles to understand emotions? "),
        yn("Difficulty maintaining conversations? "),
        yn("Does not respond to name? "),
        yn("Delayed speech development? "),
        yn("Repeats words or phrases frequently? "),
        yn("Difficulty expressing needs verbally? "),
        yn("Displays repetitive movements? "),
        yn("Becomes upset when routines change? ")
    ]

    for i in range(10):
        answers[f"A{i+1}_Score"] = aq[i]

    print("\n--- Development Milestones ---")

    fw = float(input("Age at first words (years): ") or 1)
    tw = float(input("Age at two-word phrases (years): ") or 2)
    eye = float(input("Age when consistent eye contact developed (years): ") or 0.5)
    smile = float(input("Age of social smile (years): ") or 0.5)

    answers["first_words_months"] = fw * 12
    answers["two_word_phrases_months"] = tw * 12
    answers["eye_contact_age_months"] = eye * 12
    answers["social_smile_months"] = smile * 12

    print("\n--- Behavioral Severity ---")

    answers["social_communication_delay"] = float(
        input("Social communication difficulty (0-10): ") or 0
    )

    answers["repetitive_behaviors_score"] = float(
        input("Repetitive behavior severity (0-10): ") or 0
    )

    answers["sensory_sensitivities"] = yn(
        "Sensitive to sounds, lights or textures? "
    )

    print("\n--- Medical History ---")

    answers["family_history_asd"] = yn(
        "Family history of autism? "
    )

    answers["jaundice"] = yn(
        "Jaundice at birth? "
    )

    answers["gender"] = 1

    # Defaults
    answers["used_app_before"] = 0
    answers["preeclampsia"] = 0
    answers["birth_asphyxia"] = 0
    answers["preterm_birth"] = 0
    answers["low_birth_weight"] = 0
    answers["consanguineous_marriage"] = 0
    answers["intellectual_disability"] = 0
    answers["epilepsy"] = 0
    answers["adhd"] = 0
    answers["language_disorder"] = 0
    answers["motor_delay"] = 0

    vector = build_feature_vector_from_answers(answers)

    print(f"\n[Debug] Features used: {len(vector)}")

    return np.array(vector).reshape(1, -1)

if __name__ == "__main__":
    get_terminal_input()