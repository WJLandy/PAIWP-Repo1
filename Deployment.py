import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
file_path = '/workspaces/PAIWP-Repo1/Billionaires Statistics Dataset.csv'
df = pd.read_csv(file_path)

# Check if 'finalWorth' is a string and if so, clean it
if df['finalWorth'].dtype == object:
    df['finalWorth'] = df['finalWorth'].str.replace('[\$,]', '', regex=True)

# Convert 'finalWorth' to float if it is not already
df['finalWorth'] = df['finalWorth'].astype(float)

# Clean the 'gdp_country' column by removing dollar signs, commas, and quotes
df['gdp_country'] = df['gdp_country'].str.replace('["\$,]', '', regex=True).astype(float)

# Remove unnecessary columns and any rows with missing values
columns_to_drop = ['rank', 'personName', 'city', 'source', 'countryOfCitizenship', 'latitude_country', 'longitude_country']
df_cleaned = df.drop(columns=columns_to_drop).dropna()

# Convert categorical data to numerical data using one-hot encoding
categorical_cols = ['category', 'country', 'industries']
df_encoded = pd.get_dummies(df_cleaned, columns=categorical_cols)

# Keep only the numeric columns for the model
numeric_cols = ['age', 'gdp_country', 'cpi_country', 'cpi_change_country', 
                'gross_tertiary_education_enrollment', 'gross_primary_education_enrollment_country', 
                'life_expectancy_country', 'tax_revenue_country_country', 'total_tax_rate_country', 
                'population_country']  # Add other numeric columns as needed

df_numeric = df_encoded[numeric_cols + ['finalWorth']]

# Drop rows with any missing or infinite values
df_numeric = df_numeric.replace([np.inf, -np.inf], np.nan).dropna()

# Separate the features and the target variable
X = df_numeric.drop('finalWorth', axis=1)
y = df_numeric['finalWorth']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate and print the mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

import joblib
joblib.dump(model, 'model.pkl')
