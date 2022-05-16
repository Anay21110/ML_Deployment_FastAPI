

## Importing Libraries
import uvicorn
from fastapi import FastAPI 
from BankNotes import BankNote 
import numpy as np 
import pickle
import pandas as pd 

## Create the app object
app = FastAPI()
pickle_in = open("classifier.pkl","rb")
classifier=pickle.load(pickle_in)

## Index route, opens automatically on https://127.0.0.1:8000
@app.get('/')
def index():
    return {'message':'Hello, World'}

## Route the single parameter
@app.get('/{name}')
def get_name(name: str):
    return {'Welcome the AnayS ML model': f'{name}'}

## Expose the prediction functionality, make a prediction from the passed 
## JSON data and return the predicted Bank Note with the confidense
@app.get('/predict')
def predict_banknote(data:BankNote):
    data = data.dict()
    variance = data['variance']
    skewness = data['skewness']
    curtosis = data['curtosis']
    entropy = data['entropy']
    
    prediction = classifer.predict([[variance, skewness, curtosis, entropy]])
    
    if (prediction[0]> 0.5):
        prediction = "Fake Note"
    else:
        prediction = "Its a bank note"
        
    return {
        'prediction': prediction
    }
    
## Run the API with uvicorn
if __name__=="__main__":
    uvicorn.run(app, host='127.0.0.1', port = 8000)
    
     
            
