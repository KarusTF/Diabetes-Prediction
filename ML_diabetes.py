import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import joblib


diabetes_data = pd.read_csv("diabetes.csv")

diabetes_data = diabetes_data.fillna(value=np.nan)
diabetes_data.loc[diabetes_data['Pregnancies'] < 0, 'Pregnancies'] = 0
diabetes_data.loc[diabetes_data['Age'] < 0, 'Age'] = 0

imputer = SimpleImputer(strategy='median')
diabetes_data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI','Pregnancies','Age']] = imputer.fit_transform(diabetes_data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI','Pregnancies','Age']])


# Normalize numerical features
scaler = StandardScaler()
diabetes_data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI','Pregnancies','Age']] = scaler.fit_transform(diabetes_data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI','Pregnancies','Age']])

# Split the dataset into features and target variable
X = diabetes_data.drop('Outcome', axis=1)
y = diabetes_data['Outcome']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model on the testing set
y_pred = model.predict(X_test)

# Define hyperparameters grid for logistic regression
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}

# Perform grid search cross-validation
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Re-train the model with the optimized hyperparameters
optimized_model = grid_search.best_estimator_
optimized_model.fit(X_train, y_train)

joblib.dump(optimized_model, 'model.pkl')
