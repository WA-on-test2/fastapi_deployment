import uvicorn
from fastapi import FastAPI
import numpy as np
import pandas as pd
import pickle
from pydantic import BaseModel

app=FastAPI()

pic_model=open("lasso_model.pkl","rb")
reg=pickle.load(pic_model)
class data_attr(BaseModel):
        Sex : int
        MaritalStatus  : int 
        Age  : int
        Salary :float
        AdditionalIncome :int
        HouseOwnership :int
        HomeTelephone :int
        EducationLevel : int
        LoansOtherBanks : int 
        CorporateGuarantee :  int
        Company_01 : bool   
        Company_10: bool   

@app.post('/predict')
def predict_credit_score(data: data_attr):
    # Convert input data to dictionary
    data_dict = data.dict()
    
    # Convert boolean values to integers for model compatibility
    data_dict['Company_01'] = int(data_dict['Company_01'])
    data_dict['Company_10'] = int(data_dict['Company_10'])
    
    # Create a DataFrame with the input features
    input_df = pd.DataFrame([data_dict])
    
    # Make prediction
    prediction = reg.predict(input_df)[0]
    
    # Return prediction as is, without rounding or categorizing
    return {
        'prediction': prediction
    }
