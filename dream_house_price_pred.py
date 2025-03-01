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
y_pred=model.predict(x_test)

plt.figure(figsize=(10,5))
plt.scatter(y_test,y_pred,color = "blue",label="Actual Prices")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--", label="Perfect Prediction Line")
plt.xlabel("Amenities and Features")
plt.ylabel("Prices")
plt.legend()
plt.show()

def predict_price():
    print("\nHouse Price Prediction Questionnaire \n")
    
    user_data = []
    
    # Asking user for input values
    user_data.append(float(input("Enter Area (sq ft): ")))
    user_data.append(int(input("Enter Number of Bedrooms: ")))
    user_data.append(int(input("Enter Number of Bathrooms: ")))
    user_data.append(int(input("Enter Number of Stories: ")))
    
    # Yes/No Questions
    for col in yes_no_columns:
        response = input(f"Does the house have {col.replace('_', ' ')}? (yes/no): ").strip().lower()
        user_data.append(1 if response == "yes" else 0)
    
    user_data.append(int(input("Enter Parking Spaces: ")))

    # Furnishing Status
    furnishing = input("Enter Furnishing Status (unfurnished/semi-furnished/furnished): ").strip().lower()
    furnishing_map = {"unfurnished": 0, "semi-furnished": 1, "furnished": 2}
    user_data.append(furnishing_map.get(furnishing, 0))  # Default to "unfurnished"

    # Convert to numpy array and reshape for prediction
    user_data = np.array(user_data).reshape(1, -1)
    
    # Predict price
    predicted_price = model.predict(user_data)[0]
    
    print("\n💰 Estimated House Price: ₹{:.2f}".format(predicted_price))

# Run the function
predict_price()

