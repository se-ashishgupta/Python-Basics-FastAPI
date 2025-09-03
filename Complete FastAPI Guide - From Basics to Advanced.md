# Complete FastAPI Guide - From Basics to Advanced

## 📚 Table of Contents

1. [Installation & Setup](#installation--setup)
2. [Core Libraries Explained](#core-libraries-explained)
3. [FastAPI Fundamentals](#fastapi-fundamentals)
4. [Request Handling](#request-handling)
5. [Response Models](#response-models)
6. [Validation & Error Handling](#validation--error-handling)
7. [Authentication & Security](#authentication--security)
8. [Database Integration](#database-integration)
9. [Middleware & Dependencies](#middleware--dependencies)
10. [Testing](#testing)
11. [Deployment](#deployment)
12. [Advanced Features](#advanced-features)

---

## 📦 Installation & Setup

### Required Libraries

```bash
# Core FastAPI setup
pip install fastapi uvicorn[standard] python-dotenv

# Database support
pip install sqlalchemy alembic psycopg2-binary

# Authentication & Security
pip install python-jose[cryptography] passlib[bcrypt] python-multipart

# Testing
pip install pytest httpx

# Optional utilities
pip install pydantic[email] python-dateutil requests
```

### Project Structure

```
my_fastapi_project/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── user.py
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── auth.py
│   │   └── users.py
│   ├── database.py
│   ├── dependencies.py
│   └── config.py
├── tests/
├── .env
├── requirements.txt
└── README.md
```

---

## 🔧 Core Libraries Explained

### 1. FastAPI

```python
from fastapi import FastAPI

app = FastAPI(
    title="My API",
    description="A sample FastAPI application",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)
```

**Features:**

- Modern Python web framework
- Built on Starlette (for web) and Pydantic (for data)
- Automatic API documentation
- Type hints for validation
- High performance (comparable to Node.js and Go)

### 2. Uvicorn[standard]

```bash
# Basic usage
uvicorn main:app --reload

# Production settings
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4

# With SSL
uvicorn main:app --ssl-keyfile key.pem --ssl-certfile cert.pem
```

**What `[standard]` includes:**

- `uvloop`: Ultra fast event loop
- `httptools`: Fast HTTP parsing
- `websockets`: WebSocket support
- `watchfiles`: File watching for auto-reload

### 3. Python-dotenv

```python
from dotenv import load_dotenv
import os

load_dotenv()

# Environment variables
DATABASE_URL = os.getenv("DATABASE_URL")
SECRET_KEY = os.getenv("SECRET_KEY")
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
```

**.env file example:**

```env
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/dbname
REDIS_URL=redis://localhost:6379

# Security
SECRET_KEY=your-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# API Keys
OPENAI_API_KEY=sk-your-key-here
STRIPE_SECRET_KEY=sk_test_your-key

# Environment
DEBUG=True
ENVIRONMENT=development
```

---

## 🚀 FastAPI Fundamentals

### Basic Application

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Hello World"}

# Health check
@app.get("/health")
async def health_check():
    return {"status": "healthy"}
```

### Running the Application

```bash
# Development (with auto-reload)
uvicorn main:app --reload

# Production
uvicorn main:app --host 0.0.0.0 --port 8000

# Multiple workers
uvicorn main:app --workers 4

# Custom host and port
uvicorn main:app --host 127.0.0.1 --port 3000
```

---

## 🌐 Request Handling

### HTTP Methods

```python
@app.get("/items")           # GET - Read
@app.post("/items")          # POST - Create
@app.put("/items/{id}")      # PUT - Update (replace)
@app.patch("/items/{id}")    # PATCH - Update (partial)
@app.delete("/items/{id}")   # DELETE - Remove
@app.head("/items")          # HEAD - Headers only
@app.options("/items")       # OPTIONS - Available methods
```

### Path Parameters

```python
@app.get("/users/{user_id}")
async def get_user(user_id: int):
    return {"user_id": user_id}

# Multiple path parameters
@app.get("/users/{user_id}/posts/{post_id}")
async def get_user_post(user_id: int, post_id: int):
    return {"user_id": user_id, "post_id": post_id}

# Path with validation
from enum import Enum

class ModelName(str, Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"

@app.get("/models/{model_name}")
async def get_model(model_name: ModelName):
    return {"model_name": model_name}
```

### Query Parameters

```python
# Simple query parameters
@app.get("/items")
async def read_items(skip: int = 0, limit: int = 10):
    return {"skip": skip, "limit": limit}

# Optional query parameters
@app.get("/search")
async def search(q: str | None = None, category: str | None = None):
    if q:
        return {"query": q, "category": category}
    return {"message": "No query provided"}

# Boolean query parameters
@app.get("/products")
async def get_products(in_stock: bool = True):
    return {"in_stock_only": in_stock}

# List query parameters
@app.get("/tags")
async def get_by_tags(tags: list[str] = []):
    return {"tags": tags}
```

### Request Body with Pydantic

```python
from pydantic import BaseModel, EmailStr, Field
from datetime import datetime
from typing import Optional

class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8)
    age: int = Field(..., ge=18, le=120)
    is_active: bool = True

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    is_active: bool
    created_at: datetime

@app.post("/users", response_model=UserResponse)
async def create_user(user: UserCreate):
    # Process user creation
    return UserResponse(
        id=1,
        username=user.username,
        email=user.email,
        is_active=user.is_active,
        created_at=datetime.now()
    )
```

### File Uploads

```python
from fastapi import File, UploadFile

# Single file upload
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size": len(content)
    }

# Multiple file uploads
@app.post("/upload-multiple")
async def upload_multiple(files: list[UploadFile] = File(...)):
    return [{"filename": f.filename, "size": len(await f.read())} for f in files]

# File with form data
from fastapi import Form

@app.post("/upload-with-data")
async def upload_with_data(
    file: UploadFile = File(...),
    description: str = Form(...)
):
    return {
        "filename": file.filename,
        "description": description
    }
```

---

## 📤 Response Models

### Response Model Types

```python
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class Item(BaseModel):
    id: int
    name: str
    price: float
    created_at: datetime

# Single item response
@app.get("/items/{item_id}", response_model=Item)
async def get_item(item_id: int):
    return Item(
        id=item_id,
        name="Sample Item",
        price=29.99,
        created_at=datetime.now()
    )

# List response
@app.get("/items", response_model=List[Item])
async def get_items():
    return [
        Item(id=1, name="Item 1", price=10.0, created_at=datetime.now()),
        Item(id=2, name="Item 2", price=20.0, created_at=datetime.now())
    ]

# Optional response
@app.get("/items/{item_id}", response_model=Optional[Item])
async def get_item_optional(item_id: int):
    if item_id == 1:
        return Item(id=1, name="Found Item", price=15.0, created_at=datetime.now())
    return None
```

### Custom Response Types

```python
from fastapi import Response, JSONResponse, HTMLResponse, PlainTextResponse
from fastapi.responses import FileResponse, RedirectResponse

# JSON Response with custom status
@app.get("/custom-json")
async def custom_json():
    return JSONResponse(
        content={"message": "Custom response"},
        status_code=201,
        headers={"X-Custom-Header": "value"}
    )

# HTML Response
@app.get("/html", response_class=HTMLResponse)
async def get_html():
    return "<h1>Hello HTML</h1>"

# File download
@app.get("/download")
async def download_file():
    return FileResponse("path/to/file.pdf", filename="download.pdf")

# Redirect
@app.get("/redirect")
async def redirect():
    return RedirectResponse(url="/docs")
```

---

## ✅ Validation & Error Handling

### Pydantic Validation

```python
from pydantic import BaseModel, Field, validator, EmailStr
from typing import Optional

class UserModel(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    email: EmailStr
    age: int = Field(..., ge=0, le=150)
    password: str = Field(..., min_length=8)
    confirm_password: str

    @validator('confirm_password')
    def passwords_match(cls, v, values):
        if 'password' in values and v != values['password']:
            raise ValueError('Passwords do not match')
        return v

    @validator('name')
    def name_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('Name cannot be empty')
        return v.strip()

# Custom validation function
from pydantic import validator

def validate_phone(phone: str):
    import re
    pattern = r'^\+?1?\d{9,15}$'
    if not re.match(pattern, phone):
        raise ValueError('Invalid phone number format')
    return phone
```

### Custom Exception Handling

```python
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

# Custom exception
class ItemNotFoundError(Exception):
    def __init__(self, item_id: int):
        self.item_id = item_id

# Exception handler
@app.exception_handler(ItemNotFoundError)
async def item_not_found_handler(request: Request, exc: ItemNotFoundError):
    return JSONResponse(
        status_code=404,
        content={"message": f"Item {exc.item_id} not found"}
    )

# Validation error handler
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": exc.body}
    )

# Using custom exception
@app.get("/items/{item_id}")
async def get_item(item_id: int):
    if item_id not in items_db:
        raise ItemNotFoundError(item_id)
    return items_db[item_id]
```

### HTTP Status Codes

```python
from fastapi import status

@app.post("/users", status_code=status.HTTP_201_CREATED)
async def create_user(user: UserCreate):
    return {"message": "User created"}

@app.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(user_id: int):
    # Delete user logic
    return

# Conditional status codes
from fastapi import Response

@app.put("/items/{item_id}")
async def update_item(item_id: int, item: ItemUpdate, response: Response):
    if item_id in items_db:
        # Update existing
        response.status_code = status.HTTP_200_OK
        return {"message": "Updated"}
    else:
        # Create new
        response.status_code = status.HTTP_201_CREATED
        return {"message": "Created"}
```

---

## 🔐 Authentication & Security

### JWT Authentication

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str):
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str):
    return pwd_context.verify(plain_password, hashed_password)

# JWT token creation
SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# JWT verification
security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Protected endpoint
@app.get("/protected")
async def protected_route(current_user: str = Depends(get_current_user)):
    return {"message": f"Hello {current_user}"}
```

### API Key Authentication

```python
from fastapi import Header, HTTPException

async def verify_api_key(x_api_key: str = Header()):
    if x_api_key != "your-secret-api-key":
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return x_api_key

@app.get("/api-protected")
async def api_protected(api_key: str = Depends(verify_api_key)):
    return {"message": "Access granted"}
```

### OAuth2 with Password Flow

```python
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    # Verify user credentials
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect credentials")

    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}
```

---

## 🗄️ Database Integration

### SQLAlchemy Setup

```python
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# Database configuration
DATABASE_URL = "postgresql://user:password@localhost:5432/dbname"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database model
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

### CRUD Operations

```python
from sqlalchemy.orm import Session
from fastapi import Depends

@app.post("/users")
async def create_user(user: UserCreate, db: Session = Depends(get_db)):
    # Check if user exists
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    # Create new user
    hashed_password = hash_password(user.password)
    db_user = User(
        username=user.username,
        email=user.email,
        hashed_password=hashed_password
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@app.get("/users/{user_id}")
async def get_user(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@app.get("/users")
async def get_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    users = db.query(User).offset(skip).limit(limit).all()
    return users

@app.put("/users/{user_id}")
async def update_user(user_id: int, user_update: UserUpdate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.id == user_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")

    for field, value in user_update.dict(exclude_unset=True).items():
        setattr(db_user, field, value)

    db.commit()
    db.refresh(db_user)
    return db_user

@app.delete("/users/{user_id}")
async def delete_user(user_id: int, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.id == user_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")

    db.delete(db_user)
    db.commit()
    return {"message": "User deleted"}
```

---

## 🔄 Middleware & Dependencies

### Middleware

```python
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import time

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://myapp.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "*.example.com"]
)

# Custom middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response
```

### Dependency Injection

```python
from fastapi import Depends

# Simple dependency
def common_parameters(q: str | None = None, skip: int = 0, limit: int = 100):
    return {"q": q, "skip": skip, "limit": limit}

@app.get("/items")
async def read_items(commons: dict = Depends(common_parameters)):
    return commons

# Class-based dependency
class CommonQueryParams:
    def __init__(self, q: str | None = None, skip: int = 0, limit: int = 100):
        self.q = q
        self.skip = skip
        self.limit = limit

@app.get("/users")
async def read_users(commons: CommonQueryParams = Depends()):
    return commons

# Nested dependencies
def verify_token(token: str = Depends(oauth2_scheme)):
    # Verify token logic
    return token

def get_current_user(token: str = Depends(verify_token)):
    # Get user from token
    return user

@app.get("/me")
async def read_current_user(current_user: User = Depends(get_current_user)):
    return current_user
```

---

## 🧪 Testing

### Test Setup

```python
from fastapi.testclient import TestClient
import pytest

# test_main.py
from main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}

def test_create_user():
    response = client.post(
        "/users",
        json={
            "username": "testuser",
            "email": "test@example.com",
            "password": "testpass123"
        }
    )
    assert response.status_code == 201
    assert response.json()["username"] == "testuser"

# Test with authentication
def test_protected_endpoint():
    # Login first
    login_response = client.post(
        "/token",
        data={"username": "testuser", "password": "testpass123"}
    )
    token = login_response.json()["access_token"]

    # Use token
    response = client.get(
        "/protected",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
```

### Async Testing

```python
import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_async_endpoint():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/")
    assert response.status_code == 200
```

### Database Testing

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pytest

# Test database
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

@pytest.fixture
def setup_database():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)
```

---

## 🚀 Deployment

### Production Configuration

```python
# config.py
from pydantic import BaseSettings

class Settings(BaseSettings):
    app_name: str = "My FastAPI App"
    debug: bool = False
    database_url: str
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    class Config:
        env_file = ".env"

settings = Settings()
```

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: "3.8"
services:
  web:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/myapp
    depends_on:
      - db

  db:
    image: postgres:15
    environment:
      POSTGRES_PASSWORD: password
      POSTGRES_DB: myapp
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

### Gunicorn with Uvicorn Workers

```bash
# Install gunicorn
pip install gunicorn

# Run with multiple workers
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker

# With specific host and port
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

---

## ⚡ Advanced Features

### Background Tasks

```python
from fastapi import BackgroundTasks

def send_email(email: str, message: str):
    # Email sending logic
    print(f"Sending email to {email}: {message}")

@app.post("/send-notification")
async def send_notification(
    email: str,
    background_tasks: BackgroundTasks
):
    background_tasks.add_task(send_email, email, "Welcome!")
    return {"message": "Notification sent"}
```

### WebSockets

```python
from fastapi import WebSocket, WebSocketDisconnect

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await manager.broadcast(f"Message: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
```

### Event Handlers

```python
@app.on_event("startup")
async def startup_event():
    print("Application starting up...")
    # Initialize database connections, load ML models, etc.

@app.on_event("shutdown")
async def shutdown_event():
    print("Application shutting down...")
    # Clean up resources, close connections
```

### Sub-applications

```python
from fastapi import FastAPI

# Main app
app = FastAPI()

# Sub-application
api_v1 = FastAPI()

@api_v1.get("/users")
async def get_users():
    return {"users": []}

# Mount sub-app
app.mount("/api/v1", api_v1)
```

### Custom OpenAPI Schema

```python
from fastapi.openapi.utils import get_openapi

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="My Custom API",
        version="2.5.0",
        description="This is a very custom OpenAPI schema",
        routes=app.routes,
    )
    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
    }
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
```

---

## 🔗 Router Organization

### Using APIRouter

```python
# routers/users.py
from fastapi import APIRouter, Depends

router = APIRouter(
    prefix="/users",
    tags=["users"],
    responses={404: {"description": "Not found"}},
)

@router.get("/")
async def get_users():
    return {"users": []}

@router.get("/{user_id}")
async def get_user(user_id: int):
    return {"user_id": user_id}

# main.py
from routers import users

app.include_router(users.router)
```

---

## 🛠️ Development Commands

### Running FastAPI Applications

```bash
# Development with auto-reload
uvicorn main:app --reload

# Custom host and port
uvicorn main:app --host 0.0.0.0 --port 8080

# With environment variables
uvicorn main:app --env-file .env

# Debug mode
uvicorn main:app --reload --log-level debug

# Multiple workers (production)
uvicorn main:app --workers 4

# With SSL certificates
uvicorn main:app --ssl-keyfile key.pem --ssl-certfile cert.pem
```

### Useful Development Commands

```bash
# Generate requirements
pip freeze > requirements.txt

# Install in development mode
pip install -e .

# Run tests
pytest
pytest -v                    # Verbose
pytest --cov=app           # With coverage

# Format code
black .
isort .

# Lint code
flake8 .
mypy .

# Database migrations (with Alembic)
alembic init alembic
alembic revision --autogenerate -m "Initial migration"
alembic upgrade head
```

---

## 📊 Performance & Optimization

### Async Best Practices

```python
import asyncio
import httpx

# Use async for I/O operations
@app.get("/external-data")
async def get_external_data():
    async with httpx.AsyncClient() as client:
        response = await client.get("https://api.example.com/data")
        return response.json()

# Multiple concurrent requests
@app.get("/multiple-sources")
async def get_multiple_sources():
    async with httpx.AsyncClient() as client:
        tasks = [
            client.get("https://api1.example.com"),
            client.get("https://api2.example.com"),
            client.get("https://api3.example.com")
        ]
        responses = await asyncio.gather(*tasks)
        return [r.json() for r in responses]

# Database operations with async
@app.get("/users-async")
async def get_users_async(db: AsyncSession = Depends(get_async_db)):
    result = await db.execute(select(User))
    users = result.scalars().all()
    return users
```

### Caching

```python
from functools import lru_cache
import redis

# In-memory caching
@lru_cache(maxsize=128)
def get_expensive_computation(param: str):
    # Expensive operation
    return f"Result for {param}"

# Redis caching
redis_client = redis.Redis(host='localhost', port=6379, db=0)

@app.get("/cached-data/{key}")
async def get_cached_data(key: str):
    # Check cache first
    cached = redis_client.get(f"data:{key}")
    if cached:
        return {"data": cached.decode(), "source": "cache"}

    # Compute and cache
    data = perform_expensive_operation(key)
    redis_client.setex(f"data:{key}", 3600, data)  # Cache for 1 hour
    return {"data": data, "source": "computed"}
```

---

## 🔍 Monitoring & Logging

### Logging Setup

```python
import logging
from fastapi import Request
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    logger.info(f"Request: {request.method} {request.url}")

    response = await call_next(request)

    process_time = time.time() - start_time
    logger.info(f"Response: {response.status_code} - {process_time:.3f}s")

    return response

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "version": "1.0.0"
    }
```

### Metrics Collection

```python
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import PlainTextResponse

# Metrics
REQUEST_COUNT = Counter('requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('request_duration_seconds', 'Request duration')

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()

    response = await call_next(request)

    REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path).inc()
    REQUEST_DURATION.observe(time.time() - start_time)

    return response

@app.get("/metrics")
async def get_metrics():
    return PlainTextResponse(generate_latest())
```

---

## 🌐 API Versioning

### URL Path Versioning

```python
from fastapi import APIRouter

# Version 1
v1_router = APIRouter(prefix="/api/v1", tags=["v1"])

@v1_router.get("/users")
async def get_users_v1():
    return {"version": "v1", "users": []}

# Version 2
v2_router = APIRouter(prefix="/api/v2", tags=["v2"])

@v2_router.get("/users")
async def get_users_v2():
    return {"version": "v2", "users": [], "metadata": {}}

app.include_router(v1_router)
app.include_router(v2_router)
```

### Header Versioning

```python
from fastapi import Header

@app.get("/users")
async def get_users(api_version: str = Header(default="v1", alias="X-API-Version")):
    if api_version == "v2":
        return {"version": "v2", "users": [], "metadata": {}}
    return {"version": "v1", "users": []}
```

---

## 🔒 Security Features

### Rate Limiting

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.get("/limited")
@limiter.limit("5/minute")
async def limited_endpoint(request: Request):
    return {"message": "This endpoint is rate limited"}
```

### HTTPS Redirect

```python
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware

# Force HTTPS in production
if not settings.debug:
    app.add_middleware(HTTPSRedirectMiddleware)
```

### Security Headers

```python
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response
```

---

## 📡 Real-time Features

### Server-Sent Events (SSE)

```python
from fastapi.responses import StreamingResponse
import json
import asyncio

@app.get("/events")
async def stream_events():
    async def event_generator():
        while True:
            yield f"data: {json.dumps({'timestamp': str(datetime.now())})}\n\n"
            await asyncio.sleep(1)

    return StreamingResponse(event_generator(), media_type="text/plain")
```

### Task Queues

```python
from celery import Celery

# Celery setup
celery_app = Celery("tasks", broker="redis://localhost:6379")

@celery_app.task
def send_email_task(email: str, message: str):
    # Email sending logic
    return f"Email sent to {email}"

@app.post("/send-email")
async def send_email(email: str, message: str):
    task = send_email_task.delay(email, message)
    return {"task_id": task.id, "status": "queued"}
```

---

## 📖 Complete Example Application

### main.py

```python
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI(
    title="Complete FastAPI Example",
    description="A comprehensive FastAPI application",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
from routers import auth, users, items

app.include_router(auth.router, prefix="/api/auth", tags=["authentication"])
app.include_router(users.router, prefix="/api/users", tags=["users"])
app.include_router(items.router, prefix="/api/items", tags=["items"])

@app.get("/")
async def root():
    return {"message": "Welcome to FastAPI Complete Example"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
```

---

## 🎯 Quick Commands Reference

### Virtual Environment Commands

```bash
# Create
python -m venv venv

# Activate
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# Deactivate
deactivate

# Remove
rm -rf venv/                    # Linux/Mac
rmdir /s venv\                  # Windows
```

### pip Commands

```bash
# Package management
pip install package_name
pip install package_name==1.0.0
pip install -r requirements.txt
pip uninstall package_name
pip list
pip show package_name
pip freeze > requirements.txt

# Upgrades
pip install --upgrade package_name
python -m pip install --upgrade pip

# Development
pip install -e .
pip install package_name[dev]
```

### FastAPI Development Commands

```bash
# Run development server
uvicorn main:app --reload

# Run with custom settings
uvicorn main:app --host 0.0.0.0 --port 8080 --reload

# Run tests
pytest
pytest --cov=app
pytest -v

# Code quality
black .
isort .
flake8 .
mypy .

# Database migrations
alembic revision --autogenerate -m "Add users table"
alembic upgrade head
alembic downgrade -1
```

### Nginx Configuration Commands

```bash
# Nginx setup for FastAPI
sudo nano /etc/nginx/sites-available/myapp

# Nginx configuration example:
# server {
#     listen 80;
#     server_name your-domain.com;
#
#     location / {
#         proxy_pass http://127.0.0.1:8000;
#         proxy_set_header Host $host;
#         proxy_set_header X-Real-IP $remote_addr;
#     }
# }

# Enable site
sudo ln -s /etc/nginx/sites-available/myapp /etc/nginx/sites-enabled/
sudo nginx -t                    # Test configuration
sudo systemctl reload nginx      # Reload nginx
sudo systemctl restart nginx     # Restart nginx
sudo systemctl status nginx      # Check status

# SSL with Let's Encrypt
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

### CI/CD Commands (GitHub Actions)

```bash
# GitHub Actions workflow file (.github/workflows/test.yml)
# Basic commands for CI/CD setup:

# Local testing of workflows
act                              # Run GitHub Actions locally
act -j test                      # Run specific job

# Branch protection
git branch --set-upstream-to=origin/main main
git push --set-upstream origin main
```

### Package Management Commands

```bash
# pip-tools (Requirements management)
pip install pip-tools
pip-compile requirements.in      # Generate requirements.txt
pip-compile dev-requirements.in  # Generate dev requirements
pip-sync requirements.txt        # Sync environment

# pipdeptree (Dependency tree)
pip install pipdeptree
pipdeptree                       # Show dependency tree
pipdeptree --reverse            # Show reverse dependencies
pipdeptree --json               # JSON output

# pipreqs (Generate requirements from code)
pip install pipreqs
pipreqs .                        # Generate requirements.txt from imports
pipreqs . --force               # Overwrite existing

# pipx (Install Python apps in isolated environments)
pip install pipx
pipx install package_name
pipx list
pipx uninstall package_name
```

### Database Tools Commands

```bash
# pgAdmin (PostgreSQL GUI)
pip install pgadmin4
pgadmin4

# MySQL Workbench
# GUI tool - install from website

# DBeaver (Universal database tool)
# GUI tool - install from website

# SQLite Browser
# GUI tool for SQLite

# Database connection testing
python -c "
import sqlalchemy
engine = sqlalchemy.create_engine('your_database_url')
print('Connection successful!' if engine.connect() else 'Connection failed')
"
```

### Process Management Commands

```bash
# systemd (Linux service management)
# Create service file: /etc/systemd/system/myapp.service
sudo systemctl daemon-reload
sudo systemctl enable myapp.service
sudo systemctl start myapp.service
sudo systemctl status myapp.service
sudo systemctl stop myapp.service
sudo systemctl restart myapp.service

# Supervisor (Process control system)
pip install supervisor
echo_supervisord_conf > supervisord.conf
supervisord -c supervisord.conf
supervisorctl status
supervisorctl start myapp
supervisorctl stop myapp
supervisorctl restart myapp

# PM2 (Node.js process manager - can run Python)
npm install -g pm2
pm2 start "uvicorn main:app --host 0.0.0.0 --port 8000" --name myapp
pm2 list
pm2 restart myapp
pm2 stop myapp
pm2 logs myapp
pm2 monit
```

### Log Management Commands

```bash
# journalctl (systemd logs)
journalctl -u myapp.service      # Service logs
journalctl -u myapp.service -f   # Follow logs
journalctl --since "1 hour ago"  # Time-based filtering

# logrotate (Log rotation)
sudo nano /etc/logrotate.d/myapp
sudo logrotate -d /etc/logrotate.d/myapp  # Dry run
sudo logrotate -f /etc/logrotate.d/myapp  # Force rotation

# tail commands for log monitoring
tail -f app.log                  # Follow log file
tail -n 100 app.log             # Last 100 lines
grep "ERROR" app.log | tail -20  # Last 20 errors
```

### SSL/TLS Certificate Commands

```bash
# Let's Encrypt with Certbot
sudo apt install certbot
sudo certbot certonly --standalone -d your-domain.com
sudo certbot renew               # Renew certificates
sudo certbot certificates        # List certificates

# OpenSSL (Self-signed certificates)
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes
openssl x509 -in cert.pem -text -noout  # View certificate details

# SSL testing
openssl s_client -connect your-domain.com:443
nmap --script ssl-enum-ciphers -p 443 your-domain.com
```

### Performance Monitoring Commands

```bash
# htop (System monitoring)
htop

# iostat (I/O statistics)
iostat -x 1                      # I/O stats every second

# netstat (Network connections)
netstat -tuln                    # Show listening ports
netstat -i                       # Interface statistics

# ss (Socket statistics)
ss -tuln                         # Show listening sockets

# iftop (Network bandwidth)
sudo iftop

# vmstat (Virtual memory statistics)
vmstat 1                         # Every second

# Application performance
python -m cProfile -o profile.out main.py
python -c "import pstats; p = pstats.Stats('profile.out'); p.sort_stats('cumulative').print_stats(10)"

# Memory profiling
pip install memory-profiler
python -m memory_profiler main.py
```

### Debugging Commands

```bash
# Python debugger
python -m pdb main.py            # Start with debugger
python -c "import pdb; pdb.set_trace(); your_code_here"

# ipdb (Enhanced debugger)
pip install ipdb
python -c "import ipdb; ipdb.set_trace(); your_code_here"

# Remote debugging with debugpy
pip install debugpy
python -m debugpy --listen 5678 --wait-for-client main.py

# Profiling with py-spy
pip install py-spy
py-spy record -o profile.svg -- python main.py
py-spy top --pid PID
```

### Kubernetes Commands (if deploying to K8s)

```bash
# Basic Kubernetes commands
kubectl apply -f deployment.yaml
kubectl get pods
kubectl get services
kubectl get deployments
kubectl describe pod pod-name
kubectl logs pod-name
kubectl delete pod pod-name

# Create deployment
kubectl create deployment myapp --image=myapp:latest
kubectl expose deployment myapp --type=LoadBalancer --port=80 --target-port=8000

# Scale deployment
kubectl scale deployment myapp --replicas=3

# Rolling update
kubectl set image deployment/myapp myapp=myapp:v2
kubectl rollout status deployment/myapp
kubectl rollout undo deployment/myapp
```

### AWS CLI Commands (if using AWS)

```bash
# Install AWS CLI
pip install awscli

# Configure AWS
aws configure

# S3 operations
aws s3 ls
aws s3 cp file.txt s3://bucket-name/
aws s3 sync . s3://bucket-name/

# ECS (Elastic Container Service)
aws ecs list-clusters
aws ecs list-services --cluster cluster-name
aws ecs update-service --cluster cluster-name --service service-name --desired-count 3

# Lambda
aws lambda list-functions
aws lambda invoke --function-name my-function output.txt
```

### Development Workflow Commands

```bash
# Pre-commit hooks setup
pip install pre-commit
pre-commit install
pre-commit run --all-files

# Git hooks for quality checks
# Create .pre-commit-config.yaml:
# repos:
#   - repo: https://github.com/psf/black
#     rev: 22.3.0
#     hooks:
#       - id: black
#   - repo: https://github.com/pycqa/isort
#     rev: 5.10.1
#     hooks:
#       - id: isort

# Makefile commands (create Makefile in project root)
make install                     # Install dependencies
make test                        # Run tests
make lint                        # Run linters
make format                      # Format code
make run                         # Run application
make docker-build               # Build Docker image
make docker-run                 # Run Docker container
```

### Complete Development Environment Setup

```bash
# Complete project setup script
#!/bin/bash

# Create project structure
mkdir -p myproject/{app,tests,docs,scripts}
cd myproject

# Initialize git
git init
echo "venv/\n__pycache__/\n*.pyc\n.env\n.pytest_cache/" > .gitignore

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install fastapi uvicorn[standard] python-dotenv
pip install sqlalchemy alembic psycopg2-binary
pip install pytest httpx pytest-asyncio
pip install black isort flake8 mypy
pip install pre-commit

# Create requirements files
pip freeze > requirements.txt
cat > requirements-dev.txt << EOF
pytest
httpx
pytest-asyncio
black
isort
flake8
mypy
pre-commit
EOF

# Initialize pre-commit
pre-commit install

# Create basic app structure
cat > app/main.py << EOF
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}
EOF

# Create test file
cat > tests/test_main.py << EOF
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}
EOF

# Create .env example
cat > .env.example << EOF
DATABASE_URL=postgresql://user:password@localhost:5432/dbname
SECRET_KEY=your-secret-key-here
DEBUG=True
EOF

# Create Dockerfile
cat > Dockerfile << EOF
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

echo "Project setup complete! 🎉"
echo "Next steps:"
echo "1. source venv/bin/activate"
echo "2. uvicorn app.main:app --reload"
echo "3. Open http://localhost:8000"
```

---

## 🎯 Ultimate Command Cheatsheet

### Daily Development Workflow

```bash
# 1. Environment setup
source venv/bin/activate         # Activate virtual environment
pip install -r requirements.txt  # Install dependencies

# 2. Run application
uvicorn main:app --reload        # Development server
uvicorn main:app --host 0.0.0.0 --port 8000  # Custom host/port

# 3. Database operations
alembic upgrade head             # Apply migrations
alembic revision --autogenerate -m "Description"  # Create migration

# 4. Code quality
black .                          # Format code
isort .                          # Sort imports
flake8 .                         # Lint code
mypy .                          # Type checking

# 5. Testing
pytest                          # Run all tests
pytest -v --cov=app            # Verbose with coverage

# 6. Git operations
git add .
git commit -m "feat: add new feature"
git push origin main

# 7. Deployment
docker build -t myapp .
docker run -p 8000:8000 myapp
```

### Emergency Troubleshooting

```bash
# Check if app is running
curl http://localhost:8000/health

# Check processes
ps aux | grep uvicorn
ps aux | grep gunicorn

# Check ports
netstat -tuln | grep 8000
ss -tuln | grep 8000

# Check logs
tail -f app.log
journalctl -u myapp.service -f

# Database connection test
python -c "
from sqlalchemy import create_engine
engine = create_engine('your_database_url')
conn = engine.connect()
print('DB connection OK')
conn.close()
"

# Memory usage
ps aux --sort=-%mem | head
htop

# Disk usage
df -h
du -sh /path/to/app

# Kill processes
pkill -f uvicorn
pkill -f gunicorn
```

**🚀 Now you have EVERY command you'll ever need for FastAPI development, from basic setup to production deployment!**

---

## 🎨 API Documentation URLs

When your FastAPI app is running, these URLs are automatically available:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI JSON**: `http://localhost:8000/openapi.json`

---

## 🚨 Common Gotchas & Solutions

### 1. Import Errors

```python
# Wrong - circular imports
# from models import User  # This might cause circular import

# Right - import inside function when needed
def get_user(user_id: int):
    from models import User
    return User.query.get(user_id)
```

### 2. Async/Await Usage

```python
# Wrong - mixing sync and async
@app.get("/data")
def get_data():  # Should be async
    result = await some_async_function()  # This will fail

# Right
@app.get("/data")
async def get_data():
    result = await some_async_function()
    return result
```

### 3. Database Session Management

```python
# Wrong - not closing sessions
@app.get("/users")
def get_users():
    db = SessionLocal()
    users = db.query(User).all()
    return users  # Session never closed!

# Right - using dependency injection
@app.get("/users")
def get_users(db: Session = Depends(get_db)):
    return db.query(User).all()  # Session automatically closed
```

---

## 🔥 Performance Tips

1. **Use async/await for I/O operations**
2. **Implement proper database connection pooling**
3. **Add caching for expensive computations**
4. **Use background tasks for non-critical operations**
5. **Implement proper pagination for large datasets**
6. **Use response models to control serialization**
7. **Configure appropriate worker processes in production**
8. **Monitor and log performance metrics**

---

**🎯 This guide covers everything you need to build production-ready FastAPI applications!**
