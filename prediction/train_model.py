# import os
# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# import joblib

# # Load your dataset
# # Example: insurance_data.csv with columns ['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges']
# data = pd.read_csv('insurance_data.csv')

# # Data preprocessing (convert categorical features to numerical)
# data['sex'] = data['sex'].map({'male': 1, 'female': 0})
# data['smoker'] = data['smoker'].map({'yes': 1, 'no': 0})
# data = pd.get_dummies(data, columns=['region'], drop_first=True)

# # Features and target
# X = data.drop('charges', axis=1)
# y = data['charges']

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train the model (RandomForestRegressor in this case)
# model = RandomForestRegressor(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)


# # Save the trained model to a .pkl file using joblib
# os.makedirs('prediction/model', exist_ok=True)
# joblib.dump(model, 'prediction/model/insurance_model.pkl')

# print("Model saved successfully!")

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Load your dataset
data = pd.read_csv("D:/medcostpred/insurance.csv")

# Data preprocessing (convert categorical features to numerical)
data['sex'] = data['sex'].map({'male': 1, 'female': 0})
data['smoker'] = data['smoker'].map({'yes': 1, 'no': 0})
data = pd.get_dummies(data, columns=['region'], drop_first=True)

# Features and target
X = data.drop('charges', axis=1)
y = data['charges']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model (RandomForestRegressor in this case)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model and scaler
joblib.dump(model, 'prediction/model/insurance_model.pkl')
joblib.dump(scaler, 'prediction/model/scaler.pkl')

print("Model and scaler saved successfully!")


