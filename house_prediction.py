import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

print("===== HOUSE PRICE PREDICTION PROJECT =====")

# Load Dataset
df = pd.read_csv("house_data.csv")

print("\nDataset:")
print(df)

# Check Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Dataset Information
print("\nDataset Information:")
print(df.describe())

# Visualization 1 - Scatter Plot
plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x="Area", y="Price")
plt.title("Area vs House Price")
plt.show()

# Features and Target
X = df[["Area", "Bedrooms", "Age"]]
y = df["Price"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel Trained Successfully")

# Prediction on Test Data
y_pred = model.predict(X_test)

# Accuracy
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\nModel Evaluation")
print("R2 Score:", round(r2, 2))
print("Mean Absolute Error:", round(mae, 2))

# New House Prediction
new_house = pd.DataFrame({
    "Area": [2400],
    "Bedrooms": [5],
    "Age": [3]
})

predicted_price = model.predict(new_house)

print("\nPredicted House Price")
print("₹", round(predicted_price[0]))

# Actual vs Predicted Graph
plt.figure(figsize=(8,5))
plt.plot(y_test.values, marker='o', label="Actual Price")
plt.plot(y_pred, marker='o', label="Predicted Price")
plt.title("Actual vs Predicted House Price")
plt.legend()
plt.show()

print("\nProject Completed Successfully")