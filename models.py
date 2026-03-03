from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from database import Base
import datetime

class Incident(Base):
    __tablename__ = "incidents"

    id = Column(Integer, primary_key=True, index=True)
    time = Column(DateTime, default=datetime.datetime.utcnow)
    type = Column(String)
    reason = Column(String)
    confidence = Column(Float)
    severity = Column(String)
    risk_score = Column(Float)
    lat = Column(Float)
    lng = Column(Float)
    root_causes = Column(JSON)
    speed = Column(Float)