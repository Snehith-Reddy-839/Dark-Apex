import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("anemia_dataset.csv")

# If Gender is already 0 and 1, DO NOT map.
# If Gender is Male/Female, uncomment below line:
# data["Gender"] = data["Gender"].map({"Male": 1, "Female": 0})

X = data[["Gender", "Hemoglobin", "MCH", "MCHC", "MCV"]]
y = data["Result"]   # Make sure your column name is correct

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test_scaled))
print("Model Accuracy:", accuracy)

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))