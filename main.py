# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 17:00:47 2024

@author: khaws
"""


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import json

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class model_input(BaseModel):

    Crop:str
    Season:str
    State:str
    Area:float
    Annual_Rainfall:float 
    Fertilizer:float
    Pesticide:float
    


recommendation_model = pickle.load(open('yield_prediction_trained.sav', 'rb'))
processor = pickle.load(open('processor.sav', 'rb'))

@app.post('/predict')
def yield_pred(input_parameters: model_input):
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)
    crop = input_dictionary['Crop']
    season = input_dictionary['Season']
    state = input_dictionary['State']
    area = input_dictionary['Area']
    rainfall = input_dictionary['Annual_Rainfall']
    fertilizer = input_dictionary['Fertilizer']
    pesticide = input_dictionary['Pesticide']
    


    input_list = [[crop,season,state,area,rainfall,fertilizer,pesticide]]
    transformed_input = processor.transform(input_list)
    prediction = recommendation_model.predict(transformed_input)
    
    return prediction[0]