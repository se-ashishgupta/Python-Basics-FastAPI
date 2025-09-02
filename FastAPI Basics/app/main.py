from dotenv import load_dotenv
import os
from pydantic import BaseModel, EmailStr, Field, field_validator
from typing import Optional

load_dotenv()
db_url = os.getenv("DATABASE_URL")
print(db_url)

from fastapi import FastAPI, HTTPException
app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hello World"}


# path parameter
@app.get("/items/{item_id}")
def get_item(item_id: int):
    return {"item_id": item_id}

# query parameter
@app.get("/search")
def search_item(q: str, limit: int = 10):
    return {"query": q, "limit": limit}

@app.get("/search/{user_id}/posts") # path parameter & query parameter
def get_user_posts(user_id: int, published: bool = True):
    return {"user_id": user_id, "published_only": published}

# GET request (fetch data)
@app.get("/users")
def get_users():
    return {"users": ["Alice", "Bob", "Charlie"]}


class User(BaseModel):
    name: str = Field(..., min_length=3, max_length=50)
    age: int = Field(..., gt=0, lt=100) # gt = greater than, lt = less than
    email: EmailStr
    city: Optional[str] = "No Bio Provided"

@app.post("/users")
def create_user(user: User):
    return {"message": "User created", "data": user}


class UserRegister(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8, max_length=50)
    email: EmailStr

class UserRegisterOut(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr


 # example of using field_validator to validate the password Custom Validator
# class User(BaseModel):
#     username: str
#     password: str

#     @field_validator("password")
#     def password_strength(cls, v):
#         if len(v) < 6:
#             raise ValueError("Password must be at least 6 characters long")
#         return v

@app.post("/register", response_model=UserRegisterOut) # response_model_exclude = exclude the password from the response
def register_user(user: UserRegister):
    return user

# @app.post("/register", response_model=UserRegisterOut, response_model_exclude={"password"}) # response_model_exclude = exclude the password from the response
# def register_user(user: UserRegister):
#     return user


@app.put("/users/{user_id}")
def update_user(user_id: int, user: User):
    return {"message": "User updated", "user_id": user_id, "data": user}

@app.post("/register/{user_id}/notify")
def notify_user(user_id: int, user: User, urgent: bool = False):
    return {
        "user_id": user_id,
        "user": user,
        "urgent": urgent
    }

class UserOut(BaseModel):
    username: str
    email: str

@app.get("/user-list", response_model=list[UserOut])
def get_users():
    return [
        {"username": "ashish", "password": "secret123", "email": "ashish@example.com"},
        {"username": "gupta", "password": "topsecret", "email": "gupta@example.com"},
    ]



# Error Handling
items = {"1": "Laptop", "2": "Phone"}

@app.get("/item/{item_id}")
def read_item(item_id: str):
    if item_id not in items:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"item": items[item_id]}
