import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

# Load the dataset
file_path = '/workspaces/PAIWP-Repo1/Billionaires Statistics Dataset.csv'
df = pd.read_csv(file_path)

# Data Preparation
df_cleaned = df.drop(columns=['rank', 'personName', 'city', 'source', 'countryOfCitizenship', 'latitude_country', 'longitude_country']).dropna()
df_encoded = pd.get_dummies(df_cleaned, columns=['category', 'country', 'industries'])
X = df_encoded.drop('finalWorth', axis=1)
y = df_encoded['finalWorth']

# Model Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
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