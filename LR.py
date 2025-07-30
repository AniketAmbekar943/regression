import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


df = pd.read_csv('D:/vscode/regression/datalr.csv')
# print(df)
#print(df.corr(numeric_only=True))
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

# Convert 'course_rating' to numeric
df_clean['course_rating'] = df_clean['course_rating'].str.extract(r'(\d+\.\d+)').astype(float)

# Drop rows with missing course_rating after conversion
df_clean = df_clean.dropna(subset=['course_rating'])

# Example feature: length of course_detail
df_clean['course_detail_len'] = df_clean['course_detail'].str.len()

# Define X and y
X = df_clean[['course_detail_len']]
y = df_clean['course_rating']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
avg1 = df_clean['course_detail_len'].mean()
avg2 = df_clean['course_rating'].mean()
print('Average property Price:', avg1 )
print('Average Loan Sanction Amount:', avg2)
print('R-squared:', model.score(X_test, y_test))

# -------------------------------
# Plot Actual vs Predicted

plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual', alpha=0.3)
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')

plt.xlabel('Course Detail Length')
plt.ylabel('Course Rating')
plt.title('Linear Regression Fit')
plt.legend()
plt.show()