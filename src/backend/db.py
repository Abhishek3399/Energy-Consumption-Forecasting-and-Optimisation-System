from __future__ import annotations

from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

from ..config import settings

engine = create_engine(settings.database_url, connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class EnergyRecord(Base):
    __tablename__ = "energy_records"
    id = Column(Integer, primary_key=True, index=True)
    building_id = Column(Integer, index=True)
    timestamp = Column(DateTime, index=True)
    energy_kwh = Column(Float)
    temperature_c = Column(Float)
    occupancy = Column(Float)
    is_holiday = Column(Integer)
    price_per_kwh = Column(Float)


class ForecastRecord(Base):
    __tablename__ = "forecast_records"
    id = Column(Integer, primary_key=True, index=True)
    building_id = Column(Integer, index=True)
    timestamp = Column(DateTime, index=True)
    horizon_hours = Column(Integer)
    model_type = Column(String)
    forecast_kwh = Column(Float)


class OptimizationRecord(Base):
    __tablename__ = "optimization_records"
    id = Column(Integer, primary_key=True, index=True)
    building_id = Column(Integer, index=True)
    timestamp = Column(DateTime, index=True)
    horizon_hours = Column(Integer)
    strategy = Column(String)
    optimized_load_kwh = Column(Float)
    original_cost = Column(Float)
    expected_cost = Column(Float)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

