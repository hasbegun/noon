# app/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from router import router as api_router # Import the router
from logger_config import setup_logging

from storage_service import initialize_database

setup_logging()

app = FastAPI(title="Image Analysis API")

@app.on_event("startup")
def startup_event():
    """
    Initializes the database tables and ensures the 'system' user exists.
    This must run in the main thread before any background thread accesses the DB.
    """
    initialize_database()

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For development, allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the API router
app.include_router(api_router)

@app.get("/", tags=["Health Check"])
def read_root():
    """A simple health check endpoint."""
    return {"status": "ok", "message": "Image Analysis API is running!"}
