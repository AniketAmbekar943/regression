#imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#data cleaning and logistic regression

df = pd.read_csv('D:/vscode/regression/flight.csv') #modified path
#print(df.isnull().sum())
#print(df.describe())
df_clean = df.dropna().drop_duplicates()
df_clean['Frequent'] = (df_clean['FLIGHT_COUNT'] > 10).astype(int)
# print(df.isnull().sum())
# print(df.describe())
X = df_clean[['SEG_KM_SUM']] 
y = df_clean['Frequent']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
X_fit = np.linspace(X['SEG_KM_SUM'].min(), X['SEG_KM_SUM'].max(), 500).reshape(-1, 1)
y_prob = model.predict_proba(X_fit)[:, 1]

#graph

plt.figure(figsize=(10, 6))
plt.scatter(X_test['SEG_KM_SUM'], y_test, color='skyblue', alpha=0.6, label='Actual')
plt.plot(X_fit, y_prob, color='orange', linewidth=3, label='Logistic Curve')
plt.xlabel('SEG_KM_SUM')
plt.ylabel('Frequent Flyer Probability')
plt.title('Logistic Regression Fit')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
