from pydantic import BaseModel
from typing import List
from datetime import datetime

class IncidentBase(BaseModel):
    type: str
    reason: str
    confidence: float
    severity: str
    risk_score: float
    lat: float
    lng: float
    root_causes: List[str]
    speed: float

class IncidentResponse(IncidentBase):
    id: int
    time: datetime

    class Config:
        orm_mode = True