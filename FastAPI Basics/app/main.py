from dotenv import load_dotenv
import os
from pydantic import BaseModel, EmailStr, Field, field_validator
from typing import Optional
from routers.user import router as user_router
from database import Base, engine
from fastapi import FastAPI

# Load environment variables
load_dotenv()

# Create DB tables
# Base.metadata.create_all(bind=engine)

# Create FastAPI app
app = FastAPI()

# Root route
@app.get("/")
def root():
    return {"message": "Hello World"}

# Include routers
app.include_router(user_router)