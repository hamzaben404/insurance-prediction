# src/api/main.py
import logging
import os
import time

# Sentry Imports
import sentry_sdk
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.starlette import StarletteIntegration

from src.api.routers import health, monitoring, prediction

# --- Configure Logging First ---
# It's good practice to set up your logging before other libraries initialize.
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)  # Define your application's logger

# --- Configure Sentry SDK ---
SENTRY_DSN = os.getenv("SENTRY_DSN")
RAILWAY_ENVIRONMENT = os.getenv("RAILWAY_ENVIRONMENT", "development")
APP_VERSION = os.getenv("APP_VERSION", "0.1.0")  # You can update this via env var or bumpversion

if SENTRY_DSN:
    try:
        sentry_sdk.init(
            dsn=SENTRY_DSN,
            integrations=[
                StarletteIntegration(),
                FastApiIntegration(),
            ],
            traces_sample_rate=float(
                os.getenv("SENTRY_TRACES_SAMPLE_RATE", "1.0")
            ),  # Allow configuring sample rates
            profiles_sample_rate=float(os.getenv("SENTRY_PROFILES_SAMPLE_RATE", "1.0")),
            environment=RAILWAY_ENVIRONMENT,
            release=APP_VERSION,
            send_default_pii=True,  # Set to True if you want to send PII (personally identifiable information)
        )
        logger.info(
            f"Sentry initialized for environment: {RAILWAY_ENVIRONMENT}, release: {APP_VERSION}"
        )
    except Exception as e:
        logger.error(f"Failed to initialize Sentry: {e}", exc_info=True)
else:
    logger.warning("SENTRY_DSN not found. Sentry will not be initialized.")

# --- Create FastAPI App ---
app = FastAPI(
    title="Insurance Prediction API",
    description="API for predicting vehicle insurance purchase propensity",
    version=APP_VERSION,  # Use the APP_VERSION variable
    # You can add other useful metadata here
    # openapi_url="/api/v1/openapi.json",
    # docs_url="/docs",
    # redoc_url="/redoc"
)

# --- Add Middleware ---
# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins - consider restricting in real production
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# Request Logging and Timing Middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    logger.info(f"Request started: {request.method} {request.url.path}")

    response = await call_next(request)

    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.4f}"  # Ensure it's a string
    logger.info(
        f"Request finished: {request.method} {request.url.path} - Status {response.status_code} - Processed in {process_time:.4f} seconds"
    )
    return response


# --- Include Routers ---
app.include_router(
    prediction.router, prefix="/predictions", tags=["Predictions"]
)  # Added prefix and specific tag
app.include_router(
    health.router, tags=["Health Checks"]
)  # health.router already has "Health Checks" tag
app.include_router(monitoring.router)  # monitoring.router likely has its own prefix and tags


# --- Exception Handlers ---
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    # Log the full exception details for debugging
    logger.error(
        f"Unhandled exception for request {request.method} {request.url.path}: {exc}", exc_info=True
    )
    # Sentry should capture this automatically if initialized
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An unexpected error occurred. Please try again later.",
            "error_id": str(getattr(exc, "sentry_event_id", "N/A")),
        },
    )


# --- Root Endpoint ---
@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information and documentation links."""
    return {
        "message": "Welcome to the Insurance Prediction API!",
        "version": app.version,  # Use app.version which is set from APP_VERSION
        "documentation_swagger": app.docs_url or "/docs",  # Use configured docs_url
        "documentation_redoc": app.redoc_url or "/redoc",  # Use configured redoc_url
    }


# --- Uvicorn Runner ---
if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.getenv("API_PORT", "8000"))  # Ensure default is a string if getenv returns None
    host = os.getenv("API_HOST", "0.0.0.0")
    reload_status = os.getenv("API_RELOAD", "False").lower() in ("true", "1", "t")

    logger.info(f"Starting Uvicorn server on {host}:{port} with reload: {reload_status}")
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=reload_status,
        log_level=LOG_LEVEL.lower(),  # Pass log level to uvicorn
    )
