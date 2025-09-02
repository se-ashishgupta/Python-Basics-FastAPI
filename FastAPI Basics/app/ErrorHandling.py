from fastapi import FastAPI, HTTPException
from fastapi import Request 
from fastapi.middleware.cors import CORSMiddleware
from core.logging import log_requests
import time

app = FastAPI()

app.add_middleware(
    CORSMiddleware, # CORS Middleware
    allow_origins=["*"],  # ["http://localhost:3000"] for specific frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.middleware("http")(log_requests)


# Error Handling
items = {"1": "Laptop", "2": "Phone"}

@app.get("/item/{item_id}")
def read_item(item_id: str):
    if item_id not in items:
        # raise HTTPException(status_code=404, detail="Item not found")
        raise HTTPException(status_code=404, detail={"error": "Invalid user ID", "hint": "ID must be positive"})
    return {"item": items[item_id]}



# 🔹 Global Exception Handler
from fastapi.responses import JSONResponse
from fastapi.requests import Request

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": "Something went wrong!", "detail": str(exc)},
    )
