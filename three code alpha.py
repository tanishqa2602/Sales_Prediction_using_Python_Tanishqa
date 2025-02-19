#Step 1: Import needed library

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Create a synthetic car price dataset
data = {
    "Brand": ["Toyota", "Ford", "BMW", "Audi", "Mercedes", "Honda", "Hyundai", "Chevrolet", "Nissan", "Kia"],
    "Model": ["Corolla", "Focus", "320i", "A4", "C-Class", "Civic", "Elantra", "Cruze", "Altima", "Forte"],
    "Year": [2015, 2017, 2020, 2019, 2018, 2016, 2017, 2015, 2018, 2021],
    "Mileage (km/l)": [15.0, 12.5, 10.0, 9.0, 8.5, 13.0, 14.0, 11.5, 10.5, 12.0],
    "Horsepower (HP)": [132, 160, 180, 220, 250, 140, 150, 170, 200, 160],
    "Features": [5, 4, 6, 7, 8, 5, 6, 4, 6, 7],
    "Price (USD)": [15000, 17000, 35000, 45000, 55000, 16000, 18000, 17000, 22000, 23000]
}

# Convert the dictionary to a DataFrame
df = pd.DataFrame(data)

# Save the dataset to a CSV file
df.to_csv('car_price_prediction.csv', index=False)

# Display the dataset
print("Car Price Prediction Dataset:")
print(df)

# Step 2: Load the dataset
df = pd.read_csv('car_price_prediction.csv')

# Step 3: Preprocessing - Convert categorical features to numerical ones (Brand and Model)
df['Brand'] = df['Brand'].astype('category').cat.codes
df['Model'] = df['Model'].astype('category').cat.codes

# Step 4: Split data into features (X) and target (y)
X = df.drop(columns=['Price (USD)'])
y = df['Price (USD)']

# Step 5: Split the data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 7: Make predictions on the test data
y_pred = model.predict(X_test)

# Step 8: Evaluate the model
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R-Squared: {r2_score(y_test, y_pred):.2f}")

# Step 9: Visualize the results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)
plt.title("Actual vs Predicted Car Prices")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.grid(True)
plt.show()
