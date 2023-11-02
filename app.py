from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline


application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            Cough_symptoms=bool(request.form.get('Cough_symptoms')),
            Fever=bool(request.form.get('Fever')),
            Sore_throat=bool(request.form.get('Sore_throat')),
            Shortness_of_breath=bool(request.form.get('Shortness_of_breath')),
            Headache=bool(request.form.get('Headache')),
            Age_60_above=request.form.get('Age_60_above'),
            Sex=request.form.get('Sex'),
            Known_contact=request.form.get('Known_contact')

        )

        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0")   