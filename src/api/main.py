# src/api/main.py
import logging
import os
import time

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routers import health, prediction

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Insurance Prediction API",
    description="API for predicting vehicle insurance purchase propensity",
    version="0.1.0",  # This will be updated by bumpversion
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Include routers
app.include_router(prediction.router)
app.include_router(health.router)


# Add middleware for request logging and timing
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    # Process the request
    response = await call_next(request)

    # Calculate processing time
    process_time = time.time() - start_time
    logger.info(
        f"Request {request.method} {request.url.path} processed in {process_time:.4f} seconds"
    )

    # Add processing time header
    response.headers["X-Process-Time"] = str(process_time)
    return response


# Error handler
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "message": str(exc)},
    )


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Insurance Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
    }


if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.getenv("API_PORT", 8080))

    # Run the application
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=port, reload=True)  # nosec B104
