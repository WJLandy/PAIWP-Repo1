import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np

# Load the dataset
file_path = '/workspaces/PAIWP-Repo1/Billionaires Statistics Dataset.csv'
df = pd.read_csv(file_path)

# Print initial dataframe to inspect
print("Initial Dataframe:")
print(df.head())

# Data Preparation
# Remove unnecessary columns and handle missing values
columns_to_drop = ['rank', 'personName', 'city', 'source', 'countryOfCitizenship',
                   'latitude_country', 'longitude_country', 'organization', 'status', 
                   'gender', 'birthDate', 'lastName', 'firstName', 'title', 'date', 
                   'state', 'residenceStateRegion']
df_cleaned = df.drop(columns=columns_to_drop).dropna()

# Handle currency formatting in gdp_country
df_cleaned['gdp_country'] = df_cleaned['gdp_country'].replace('[\$,]', '', regex=True).astype(float)

# Ensure all categorical data is properly encoded using one-hot encoding
categorical_columns = ['category', 'country', 'industries']
df_encoded = pd.get_dummies(df_cleaned, columns=categorical_columns)

# Print the cleaned dataframe to verify changes
print("Dataframe after cleaning and encoding:")
print(df_encoded.head())

# Separate the features and the target variable
X = df_encoded.drop('finalWorth', axis=1)
y = df_encoded['finalWorth']

# Convert boolean columns to integers
for column in X.select_dtypes(include=['bool']).columns:
    X[column] = X[column].astype(int)

# Print the feature matrix and target vector to inspect
print("Features (X):")
print(X.head())
print("Target (y):")
print(y.head())

# Check for multicollinearity using VIF
def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

vif_data = calculate_vif(X)
print("VIF before dropping features:")
print(vif_data)

# Drop features with VIF > 10
high_vif_columns = vif_data[vif_data["VIF"] > 10]["feature"]
X = X.drop(columns=high_vif_columns)

vif_data_after = calculate_vif(X)
print("VIF after dropping features:")
print(vif_data_after)

# Ensure all data is numeric
print("Data types of features:")
print(X.dtypes)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Hyperparameter Tuning
param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}
ridge_model = Ridge()
grid_search = GridSearchCV(estimator=ridge_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
best_ridge_model = grid_search.best_estimator_

# Evaluate the model
y_pred = best_ridge_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")