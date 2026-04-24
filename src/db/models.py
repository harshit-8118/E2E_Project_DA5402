# src/db/models.py
# Pydantic schemas for MongoDB documents

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, EmailStr, Field


#  user
class UserCreate(BaseModel):
    username : str
    email    : EmailStr
    password : str
    gender   : str = "prefer_not_to_say"   # male | female | other | prefer_not_to_say


class UserLogin(BaseModel):
    email    : EmailStr
    password : str


class UserPublic(BaseModel):
    uid              : str
    username         : str
    email            : str
    gender           : str
    verified         : bool
    created_at       : str
    prediction_count : int = 0
    feedback_count   : int = 0


#  OTP
class OTPRequest(BaseModel):
    email : EmailStr


class OTPVerify(BaseModel):
    email : EmailStr
    otp   : str


#  toke
class Token(BaseModel):
    access_token : str
    token_type   : str = "bearer"
    uid          : str
    username     : str
    email        : str


#  prediction 
class PredictionRecord(BaseModel):
    prediction_id  : str
    uid            : str
    username       : str
    predicted_class: str
    confidence     : float
    risk_level     : str
    all_scores     : dict
    inference_ms   : float
    image_id       : Optional[str] = None   # ref to images collection
    image_filename : Optional[str] = None
    timestamp      : str


#  feedback 
class FeedbackRecord(BaseModel):
    prediction_id : str
    uid           : str
    username      : str
    vote          : str
    comment       : str = ""
    timestamp     : str


#  image
class ImageRecord(BaseModel):
    image_id      : str
    prediction_id : str
    uid           : str
    username      : str
    filename      : str
    image_b64     : str   # base64 encoded — used for future retraining
    image_size_kb : float
    timestamp     : str