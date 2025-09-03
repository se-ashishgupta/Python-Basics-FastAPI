# app/routers/user.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from models import User, Item
from schemas import UserCreate, UserResponse, ItemCreate, ItemResponse
from dependencies import get_db
from uuid import UUID

router = APIRouter(prefix="/users", tags=["Users"])

@router.post("/", response_model=UserResponse)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = User(name=user.name, email=user.email, password=user.password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@router.get("/{user_id}", response_model=UserResponse)
def get_user(user_id: UUID, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@router.post("/{user_id}/items", response_model=ItemResponse)
def create_item(user_id: UUID, item: ItemCreate, db: Session = Depends(get_db)):
    db_item = Item(title=item.title, description=item.description, user_id=user_id)
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item