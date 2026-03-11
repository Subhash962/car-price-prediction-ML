
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor

print("Loading dataset...")

data = pd.read_csv("Car details v3.csv")

# Clean columns
data['engine'] = data['engine'].str.replace(' CC','', regex=False)
data['max_power'] = data['max_power'].str.replace(' bhp','', regex=False)

data['engine'] = pd.to_numeric(data['engine'], errors='coerce')
data['max_power'] = pd.to_numeric(data['max_power'], errors='coerce')

data = data.dropna()

X = data[['km_driven','engine','max_power']]
y = data['selling_price']

print("Training model...")

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X,y)

pickle.dump(model, open("model.pkl","wb"))

print("Model saved as model.pkl")
