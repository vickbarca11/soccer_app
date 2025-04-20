from fastapi import FastAPI, Query
from pydantic import BaseModel, Field
from typing import Optional
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware


from preprocessing_utils import clean_categories

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or restrict to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/models")
def get_models():
    return {
        "models": [
            {"name": "epl_goalsmodel", "description": "Predicts goal likelihood in EPL matches"},
            {"name": "messi_goalsmodel", "description": "Predicts goal likelihood for Messi"},
            {"name": "epl_outcomemodel", "description": "Predicts EPL match outcomes"}
        ]
    }

# Load model for EPL matches
model_eplmatches = joblib.load("eplmatches5ymodel_rf.pkl")

# Pydantic model
class eploutcomedata(BaseModel):
    position_away: float
    position_home: float
    match_temperature: float
    wind_speed: float	
    humidity: float
    pressure: float
    clouds: float
    team_name_home: str
    team_name_away: str
    time_of_day: str

# Model wrapper
def epl_outcomemodel(position_away, position_home, match_temperature, wind_speed, humidity, pressure, clouds, team_name_home, team_name_away, time_of_day):

    columns = ['position_away', 'position_home', 'match_temperature', 'wind_speed', 'humidity', 'pressure', 'clouds', 'team_name_home', 'team_name_away', 'time_of_day']
    features = [position_away, position_home, match_temperature, wind_speed, humidity, pressure, clouds, team_name_home, team_name_away, time_of_day]

    df = pd.DataFrame([features], columns=columns)
    df=clean_categories(df)
    
    prediction = model_eplmatches.predict_proba(df)[0][1]
    return prediction

# Prediction route for EPL match outcomes
@app.post("/predict/matchoutcome/epl")
def predict_eplmatchoutcome(data: eploutcomedata):
    prediction = epl_outcomemodel(data.position_away, data.position_home, data.match_temperature, data.wind_speed, data.humidity, data.pressure, data.clouds, data.team_name_home, data.team_name_away, data.time_of_day)
    
    return {"prediction": prediction}





# Load model for La Liga matches
model_laligamatches = joblib.load("laligamatches5ymodel_rf.pkl")

# Pydantic model
class laligaoutcomedata(BaseModel):
    position_away: float
    position_home: float
    match_temperature: float
    wind_speed: float	
    humidity: float
    pressure: float
    clouds: float
    team_name_home: str
    team_name_away: str
    time_of_day: str

# Model wrapper
def laliga_outcomemodel(position_away, position_home, match_temperature, wind_speed, humidity, pressure, clouds, team_name_home, team_name_away, time_of_day):

    columns = ['position_away', 'position_home', 'match_temperature', 'wind_speed', 'humidity', 'pressure', 'clouds', 'team_name_home', 'team_name_away', 'time_of_day']
    features = [position_away, position_home, match_temperature, wind_speed, humidity, pressure, clouds, team_name_home, team_name_away, time_of_day]

    df = pd.DataFrame([features], columns=columns)
    df=clean_categories(df)
    
    prediction = model_laligamatches.predict_proba(df)[0][1]
    return prediction





# Load model for EPL goals
model_epl = joblib.load("eplgoalsmodel_rf.pkl")

# Pydantic model
class eplgoaldata(BaseModel):
    match_period: int
    minute_in_half: int	
    possession_team: str
    play_pattern: str
    position: str
    x: float	
    y: float

# Model wrapper
def epl_goalsmodel(match_period, minute_in_half, possession_team, play_pattern, position, x,y):

    columns = ['match_period', 'minute_in_half', 'possession_team', 'play_pattern', 'position','x','y']
    features = [match_period, minute_in_half, possession_team, play_pattern, position, x, y]

    df = pd.DataFrame([features], columns=columns)
    df=clean_categories(df)
    
    prediction = model_epl.predict_proba(df)[0][1]
    return prediction

# Prediction route for EPL goals
@app.post("/predict/goals/epl")
def predict_eplgoals(data: eplgoaldata):
    prediction = epl_goalsmodel(data.match_period, data.minute_in_half, data.possession_team, data.play_pattern, data.position, data.x, data.y)
    
    return {"prediction": prediction}





# Load model for Messi goals
model_messi = joblib.load("messigoalsmodel_rf.pkl")

# Pydantic model
class messigoaldata(BaseModel):
    match_period: int
    minute_in_half: int		
    play_pattern: str
    under_pressure: bool
    x: float	
    y: float

# Model wrapper
def messi_goalsmodel(match_period, minute_in_half, play_pattern, under_pressure, x,y):

    columns = ['match_period', 'minute_in_half', 'play_pattern', 'under_pressure','x','y']
    features = [match_period, minute_in_half, play_pattern, under_pressure, x, y]

    df = pd.DataFrame([features], columns=columns)
    df=clean_categories(df)
    
    prediction = model_messi.predict_proba(df)[0][1]
    return prediction

# Prediction route for Messi goals
@app.post("/predict/goals/messi")
def predict_messigoals(data: messigoaldata):
    prediction = messi_goalsmodel(data.match_period, data.minute_in_half, data.play_pattern, data.under_pressure, data.x, data.y)
    
    return {"prediction": prediction}