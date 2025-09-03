# app/schemas.py
from pydantic import BaseModel, EmailStr
from uuid import UUID

class UserCreate(BaseModel):
    name: str
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: UUID
    name: str
    email: EmailStr

    class Config:
        from_attributes = True  # allows ORM → Pydantic conversion

class ItemCreate(BaseModel):
    title: str
    description: str

class ItemResponse(BaseModel):
    id: UUID
    title: str
    description: str

    class Config:
        from_attributes = True  # allows ORM → Pydantic conversion