from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

model = joblib.load('Mental_health_model.pkl')

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Student_Data(BaseModel):
    age:                    int = Field(..., ge=10, le=100)
    gender:                 str = Literal['Male', 'Female']
    country:                str
    academic_Level:         str = Literal['High School', 'Undergraduate', 'Graduate']
    most_used_platform:     str = Literal['Facebook','LinkedIn','Instagram','Snapchat','Twitter','YouTube','TikTok','LINE','KakaoTalk','VKontakte','WhatsApp','WeChat']
    purpose_of_use:         str = Literal['Networking', 'Education', 'Entertainment', 'News']
    avg_daily_usage_hours:  float = Field(..., ge=0, le=24)
    daily_unlocks:          int = Field(..., ge=0)
    study_hours:            int = Field(..., ge=0, le=24)
    physical_activity_hours:int = Field(..., ge=0, le=24)
    sleep_hours_per_night:  float = Field(..., ge=0, le=24)
    stress_level:           str = Literal['Low', 'Medium', 'High', 'Very High']



#Describe what we send back
class PredictionResponse(BaseModel):
   predicted_mental_health_score: float




@app.get('/')
def greet():
    return {'Welcome to Rushis School'}

top_countries = ['Other','India','USA','Canada','Australia','UK','Germany','Turkey','Mexico','France',
'Spain']

@app.post('/predict', response_model=PredictionResponse)
def predict(data: Student_Data):

    if data.country in top_countries:
        country_group = data.country
    else:
        country_group = 'Other'

    input_row = pd.DataFrame([{

        'Age':                    data.age,
        'Gender':                 data.gender,
        'Country':                data.country,
        'Academic_Level':         data.academic_Level,
        'Most_Used_Platform':     data.most_used_platform,
        'Purpose_Of_Use':         data.purpose_of_use,
        'Avg_Daily_Usage_Hours':  data.avg_daily_usage_hours,
        'Daily_Unlocks':          data.daily_unlocks,
        'Study_Hours':            data.study_hours,
        'Physical_Activity_Hours':data.physical_activity_hours,
        'Sleep_Hours_Per_Night':  data.sleep_hours_per_night,
        'Stress_Level':           data.stress_level,
        'Grouped_country':        country_group

    }])

    prediction = model.predict(input_row)[0]
    return PredictionResponse(predicted_mental_health_score=round(float(prediction), 2))