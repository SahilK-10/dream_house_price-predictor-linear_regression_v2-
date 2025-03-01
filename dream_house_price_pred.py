import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv("Housing.csv")


df["furnishingstatus"] = df["furnishingstatus"].map(
    {"unfurnished": 0, "semi-furnished": 1, "furnished": 2})
yes_no_columns = ["mainroad", "guestroom", "basement",
                  "hotwaterheating", "airconditioning"]

df[yes_no_columns] = df[yes_no_columns].apply(
    lambda x: x.map({"yes": 1, "no": 0}))


x = df[["area", "bedrooms", "bathrooms", "stories", "mainroad", "guestroom",
        "basement", "hotwaterheating", "airconditioning", "parking", "furnishingstatus"]]
y = df["price"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, color="blue", label="Actual Prices")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         color="red", linestyle="--", label="Perfect Prediction Line")
plt.xlabel("Amenities and Features")
plt.ylabel("Prices")
plt.legend()
plt.show()

print("\nHouse Price Prediction Questionnaire \n")


def predict_price():

    # User input
    user_data = [
        float(input("Enter Area (sq ft): ")),
        int(input("Enter Number of Bedrooms: ")),
        int(input("Enter Number of Bathrooms: ")),
        int(input("Enter Number of Stories: ")),
        int(input("Does the house have a main road? (1 for Yes, 0 for No): ")),
        int(input("Does the house have a guest room? (1 for Yes, 0 for No): ")),
        int(input("Does the house have a basement? (1 for Yes, 0 for No): ")),
        int(input("Does the house have hot water heating? (1 for Yes, 0 for No): ")),
        int(input("Does the house have air conditioning? (1 for Yes, 0 for No): ")),
        int(input("Enter Number of Parking Spaces: ")),
        int(input("Enter Furnishing Status (0: Unfurnished, 1: Semi-Furnished, 2: Furnished): "))
    ]

    # Convert to DataFrame with correct column names
    user_data = pd.DataFrame([user_data], columns=x.columns)

    # Predict price
    predicted_price = model.predict(user_data)[0]

    print(f"\nEstimated House Price: ₹{predicted_price:.2f}")


# Run the function
predict_price()
