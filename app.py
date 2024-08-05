from flask import Flask,render_template,request
import pandas as pd
import pickle as pkl
import numpy as np



model=pkl.load(open("LinearRegressionModel.pkl",'rb'))

app = Flask(__name__)
car = pd.read_csv("cardata.csv")

@app.route('/')
def index():
    companis = sorted(car['Make'].unique())
    car_model = sorted(car['Model'].unique())
    year = sorted(car['Year'].unique(),reverse=True)
    mileage_values = car['Mileage'].unique()  # Extract unique mileage values
    mileage_values = [int(m) for m in mileage_values]
    mileage = sorted(mileage_values)
    condition = car['Condition'].unique()

    return render_template('index.html',companis=companis,car_models=car_model,years=year,mileages=mileage,conditions=condition)

@app.route('/predict', methods=['POST'])
def predict():
    company = request.form.get('company')
    car_model = request.form.get('car_model')
    year = request.form.get('year')
    mileage = request.form.get('mileage')
    condition = request.form.get('condition')
    print(company,car_model,year,mileage,condition)
    input_data = pd.DataFrame([[company, car_model, year, mileage, condition]],
                                  columns=['Make', 'Model', 'Year', 'Mileage', 'Condition'])
    try:
        prediction = model.predict(input_data)
        predicted_price = prediction[0]  # Assuming prediction returns a list/array
    except Exception as e:
        return str(e)

    return str(np.round(predicted_price,2))
if __name__=="__main__":
    app.run(debug=True)

