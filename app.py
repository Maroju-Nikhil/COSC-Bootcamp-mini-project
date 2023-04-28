from flask import Flask, render_template
from flask import request
import pandas as pd
import numpy as np
import pickle  

car = pd.read_csv('cleaned_car.csv')
app = Flask(__name__)
model = pickle.load(open("LinearRegressionModel.pkl", "rb"))

@app.route('/')
def index():
    car_companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    years = sorted(car['year'].unique(), reverse=True)
    fuel_type = car['fuel_type'].unique()
    data = {
        "companies": car_companies,
        "models": car_models,
        "years": years,
        "fuel_types": fuel_type,
    }
    return render_template('index.html', data = data)


@app.route('/predict', methods=['POST'])
def predict():
    company = request.form.get('company')
    car_model = request.form.get('model')
    year = int(request.form.get('year'))
    fuel_type = request.form.get('fuel_type')
    kms_driven = int(request.form.get('kms_driven'))
    print(company, car_model, year, fuel_type, kms_driven)
    if not kms_driven or not company or not car_model or not fuel_type or not year: 
        return "Insufficient Details"
    
    data = pd.DataFrame([[car_model, company, year, fuel_type, kms_driven]], columns=['name','company','year','fuel_type','kms_driven'])
    prediction = model.predict(data)
    prediction = np.round(prediction[0], 2)
    print(prediction)
    return str(prediction)


if __name__ == "__main__":
    app.run(debug=False)
