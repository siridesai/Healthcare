import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

# Load data
df = pd.read_csv(r'C:\Users\njoshi\OneDrive - Eastside Preparatory School\Data Jungle\life_expectancy_data_final.csv')

# One-hot encode the 'location' column
oe = OneHotEncoder(sparse_output=False)
country_encoded = oe.fit_transform(df[['location']])  # Note: 2D input required
country_encoded_df = pd.DataFrame(country_encoded, columns=oe.get_feature_names_out(['location']))

# Merge the encoded features with original DataFrame
df = pd.concat([df[['year', 'life_expectancy']], country_encoded_df], axis=1)

# Prepare X and y
X = pd.concat([df[['year']], country_encoded_df], axis=1)
y = df['life_expectancy']

# Fit model
model = LinearRegression()
model.fit(X, y)

# Predict for a specific country and year
country_name = 'India'
year = 2022
# Build input vector manually
country_vector = [1 if name == f'location_{country_name}' else 0 for name in oe.get_feature_names_out(['location'])]
X_new = pd.DataFrame([[year] + country_vector], columns=X.columns)

# Prediction
predicted_life = model.predict(X_new)
print(f"Predicted life expectancy in {country_name} {year}: {predicted_life[0]:.2f}")

# Plot actual vs predicted for that country
mask = df[country_encoded_df.columns] == 1
mask = mask[f'location_{country_name}']  # True only for that country

df_country = df[mask]
X_country = df_country[X.columns]
y_actual = df_country['life_expectancy']
y_predicted = model.predict(X_country)

plt.figure(figsize=(8, 5))
plt.scatter(df_country['year'], y_actual, color='blue', label='Actual Data', s=50)
plt.plot(df_country['year'], y_predicted, color='red', label='Regression Line', linewidth=2)
plt.xlabel('Year')
plt.ylabel('Life Expectancy')
plt.title(f'Life Expectancy Over Time in {country_name}')
plt.legend()
plt.grid(True)
plt.show()
