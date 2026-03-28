import joblib

enc = joblib.load("models/label_encoders.pkl")

print(enc['batting_team'].classes_)