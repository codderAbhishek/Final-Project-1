from flask import Flask,render_template,request
import pandas as pd
import pickle
import sklearn

app=Flask(__name__)

model=pickle.load(open("linearregession_model.pkl",'rb'))
car=pd.read_csv('cleaned_car.csv')


@app.route('/')
def index():
    companies=sorted(car['company'].unique())
    car_model=sorted(car['name'].unique())
    year=sorted(car['year'].unique(),reverse=True)
    fuel_type=car['fuel_type'].unique()
    return render_template('index.html',companies=companies,car_model=car_model,year=year,fuel_type=fuel_type)
@app.route('/predict',methods=['POST'])
def predict():
    company=request.form.get('company')
    car_model=request.form.get('car_model')
    year=int(request.form.get('year'))
    fuel_type=request.form.get('fuel_type')
    kilo_driven=int(request.form.get('kilo_driven'))

    predicton=model.predict([[car_model,company,year,kilo_driven,fuel_type]],columns=['name','company','year','kilo_driven','fuel_type'])
    print(predicton)
    return ""


if __name__=="__main__":
    app.run(debug=True)