#!/usr/bin/env python
# coding: utf-8

# In[14]:



import pandas as pd
import datetime
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


# Open the file for reading
with open('prices.csv', 'r') as f:
    # Read the lines of the file into a list, skipping the first line
    lines = f.readlines()[1:]

# Create empty lists to hold the prices and dates
prices = []
dates = []

# Loop through the lines of the file
for line in lines:
    # Split the line into a date string and a price string
    date_str, price_str = line.strip().split(',')

    # Parse the price from the price string
    price = float(price_str)

    # Parse the date from the date string
    try:
        date = datetime.datetime.strptime(date_str, '%m/%d/%Y').strftime('%d-%b-%y')
    except ValueError:
        date = datetime.datetime.strptime(date_str, '%d-%m-%y').strftime('%d-%b-%y')

    # Add the price and date to their respective lists
    prices.append(price)
    dates.append(date)

# Create a dictionary with the prices and dates
data = {'date': dates, 'price': prices}
# Create a DataFrame from the dictionary
df = pd.DataFrame(data)
df.info()
# Print the resulting DataFrame
print(df)



# Check for missing or zero values in the 'price' column
missing_mask = df['price'].isna()
zero_mask = df['price'].isin([0])

# Count the number of missing or zero values
num_missing = missing_mask.sum()
num_zeros = zero_mask.sum()

# Print the number of missing and zero values
print(f"Number of missing values: {num_missing}")
print(f"Number of zero values: {num_zeros}")


# Calculate the number of rows in the data
num_rows = len(df)

# Calculate the index of the row that corresponds to 80% of the data
train_index = int(0.8 * num_rows)

# Select the first 80% of the rows for the training set
train_data = df.iloc[:train_index]

# Select the last 20% of the rows for the testing set
test_data = df.iloc[train_index:]

# Print the size of the training and testing sets
print(f"Number of rows in training set: {len(train_data)}")
print(f"Number of rows in testing set: {len(test_data)}")





# creat input x and output y for the training dataset
# Create empty lists to hold the x and y values
x_train_data = []
y_train_data = []

# Loop through the prices, starting from the 6th day
for i in range(5, len(train_data)-1):
    # Get the previous 5 days' prices
    x_values = train_data.iloc[i-5:i]['price'].values
    # Get the price for the next day
    y_value = train_data.iloc[i+1]['price']
    # Append x and y to their respective lists
    x_train_data.append(x_values)
    y_train_data.append(y_value)

# Convert the x and y lists to numpy arrays
x_train_data = np.array(x_train_data)
y_train_data = np.array(y_train_data)

# Print the resulting x and y arrays
print(x_train_data)
print(y_train_data)



# creat input x and output y for the testing dataset
# Create empty lists to hold the x and y values
x_test_data = []
y_test_data = []

# Loop through the prices, starting from the 6th day
for i in range(5, len(test_data)-1):
    # Get the previous 5 days' prices
    x_values = test_data.iloc[i-5:i]['price'].values
    # Get the price for the next day
    y_value = test_data.iloc[i+1]['price']
    # Append x and y to their respective lists
    x_test_data.append(x_values)
    y_test_data.append(y_value)

# Convert the x and y lists to numpy arrays
x_test_data = np.array(x_test_data)
y_test_data = np.array(y_test_data)

# Print the resulting x and y arrays
print(x_test_data)
print(y_test_data)


# Create a LinearRegression object
regressor = LinearRegression()

# Fit the regressor to the training data
regressor.fit(x_train_data, y_train_data)

# Print the coefficients of the linear model
print("Model coefficients:", regressor.coef_)

# Predict the price for the next day

y_pred = regressor.predict(x_test_data)

# Print the predicted price
print("Predicted price for the next day:", y_pred)


# Plot the actual vs. predicted values as lines with different colors
plt.plot(y_test_data, label='Actual', color='blue')
plt.plot(y_pred, label='Predicted', color='red')

# Add a legend, x-label, and y-label
plt.legend()
plt.xlabel('Time')
plt.ylabel('Price')

# Show the plot
plt.show()

print(y_test_data)
print(x_test_data)


# In[ ]:





# In[15]:


import statsmodels.api as sm
s = sm.add_constant(x_test_data)
res = sm.OLS(y_test_data, s).fit()
print(res.summary())


# # Problem #5
