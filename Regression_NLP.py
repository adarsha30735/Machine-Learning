# Regression_NLP.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load energy efficiency dataset
data = pd.read_excel('EnergyEfficiency.xlsx')
X = data[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']]
y1 = data['Y1']  # Heating Load
y2 = data['Y2']  # Cooling Load

# Split the data for regression
X_train, X_test, y_train, y_test = train_test_split(X, y1, test_size=0.2, random_state=42)

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Evaluate the regression model
print(f'Linear Regression Coefficients: {lin_reg.coef_}')
print(f'Intercept: {lin_reg.intercept_}')

# For K-Nearest Neighbors classification on text data (placeholder example)
# Assuming 'text_data' is a DataFrame containing text data for classification
text_data = pd.DataFrame({
    'text': ['example text data', 'another example'],
    'label': [0, 1]
})

# Example KNN Classification
# Convert text to numerical (this is a placeholder, typically you'd use TF-IDF or similar)
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X_text = vectorizer.fit_transform(text_data['text']).toarray()
y_text = text_data['label']

X_train_text, X_test_text, y_train_text, y_test_text = train_test_split(X_text, y_text, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_text, y_train_text)

# Evaluate the KNN classifier
accuracy_text = knn.score(X_test_text, y_test_text)
print(f'KNN Classifier Accuracy on Text Data: {accuracy_text}')
