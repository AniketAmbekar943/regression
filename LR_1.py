import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv('train.csv')
# print(df)

# Data exploration and cleaning
# df.info()
# print(df.isnull().sum())
# print(df.describe())

# Removing rows with missing values
df_clean = df.dropna()
# print(df_clean.isnull().sum())

# Handling Duplicate values
n_duplicates = df_clean.duplicated().sum()
df_clean = df_clean.drop_duplicates()
# df_clean.describe()
# df_clean.info()
# df_clean.to_csv('datalr_clean.csv', index=False)

# Simple Linear Regression



# Drop rows with missing income after conversion
df_clean = df_clean.dropna(subset=['Income (USD)'])
df_clean = df_clean.dropna(subset=['Loan Sanction Amount (USD)'])
# Define X and y
X = df_clean[['Property Price']]
y = df_clean['Loan Sanction Amount (USD)']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
avg1= df['Property Price'].mean()
avg2= df['Loan Sanction Amount (USD)'].mean()
print('Average property Price:', avg1 )
print('Average Loan Sanction Amount:', avg2)
print('R-squared:', model.score(X_test, y_test))

# -------------------------------
# Plot Actual vs Predicted

plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual', alpha=0.3)
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')

plt.xlabel('Loan Sanction Amount (USD)')
plt.ylabel('Income (USD)')
plt.title('Linear Regression Fit')
plt.legend()
plt.show()