# Car Price Prediction using Linear Regression

## Project Overview
This project implements a Linear Regression model to predict car prices based on various features such as brand, model, year, mileage, horsepower, and additional features. The dataset used is synthetic and serves as a demonstration of how regression techniques can be applied to pricing predictions.

## Features
- Data preprocessing (converting categorical data to numerical values)
- Splitting data into training and testing sets
- Training a Linear Regression model
- Evaluating model performance using MAE, MSE, and R-squared score
- Visualizing actual vs predicted prices

## Dataset
The dataset contains the following columns:
- `Brand`: Car manufacturer (e.g., Toyota, Ford, BMW, etc.)
- `Model`: Specific model of the car
- `Year`: Manufacturing year of the car
- `Mileage (km/l)`: Fuel efficiency of the car
- `Horsepower (HP)`: Engine power of the car
- `Features`: Number of additional features (e.g., sunroof, GPS, etc.)
- `Price (USD)`: The price of the car in US dollars (target variable)

## Requirements
To run this project, install the required dependencies:
```bash
pip install pandas numpy matplotlib scikit-learn
```

## Running the Analysis
1. Clone the repository:
```bash
git clone https://github.com/yourusername/car-price-prediction.git
```
2. Navigate to the project folder:
```bash
cd car-price-prediction
```
3. Run the Python script:
```bash
python three_code_alpha.py
```

## Visualizations
- Scatter plot of actual vs predicted car prices
- Regression line to visualize prediction accuracy

## Results
- The model provides car price predictions based on historical data.
- Evaluation metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R²) provide insights into model performance.
- The scatter plot visualizes the relationship between actual and predicted prices.

