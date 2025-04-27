# Vehicle Insurance Prediction API

## Overview

This is a FastAPI-based REST API for predicting whether customers will purchase vehicle insurance. The API provides endpoints for single and batch predictions, health checks, and model information.

## Features

- Single and batch prediction endpoints
- API key authentication
- Request validation with Pydantic
- Automatic API documentation
- Dockerized for easy deployment
- Health monitoring endpoints

## Quick Start

### Prerequisites

- Docker and Docker Compose installed
- Python 3.10+ (for local development)

### Running with Docker Compose

1. Clone the repository
2. Create a `.env` file with your API keys:
   ```
   API_KEYS=your-api-key-1,your-api-key-2
   ```
3. Build and run the containers:
   ```bash
   docker-compose up --build
   ```

The API will be available at `http://localhost:8000`, and the MLflow UI at `http://localhost:5000`.

### Local Development

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set environment variables:
   ```bash
   export API_KEYS=your-api-key
   export MODEL_PATH=path/to/your/model.pkl
   ```

4. Run the API:
   ```bash
   uvicorn src.api.app:app --reload
   ```

## API Endpoints

### Health Check
- **GET** `/health`
- No authentication required
- Returns API health status

### Model Information
- **GET** `/model`
- Requires API key
- Returns information about the loaded model

### Single Prediction
- **POST** `/predict`
- Requires API key
- Request body: Single customer data
- Returns prediction and probability

### Batch Prediction
- **POST** `/predict/batch`
- Requires API key
- Request body: Array of customer data
- Returns array of predictions

## Authentication

API key authentication is required for all prediction endpoints. Include your API key in the header:

```
X-API-Key: your-api-key
```

## API Documentation

Interactive API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Testing

Run the test script to verify API functionality:
```bash
python test_api.py
```

## Security Considerations

- Always use HTTPS in production
- Rotate API keys regularly
- Monitor rate limits and adjust as needed
- Keep the model file in a secure location

## Example Request

Single prediction request:
```json
{
  "gender": "Male",
  "age": 35,
  "has_driving_license": 1.0,
  "region_id": 21.0,
  "switch": 0.0,
  "vehicle_age": "1-2 Year",
  "past_accident": "No",
  "annual_premium": 15000.0,
  "sales_channel_id": 152.0,
  "days_since_created": 100.0
}
```

Example response:
```json
{
  "prediction": 1,
  "probability": 0.85,
  "request_id": "123e4567-e89b-12d3-a456-426614174000",
  "timestamp": "2024-04-20T10:30:00"
}
```

## Error Codes

- 200: Success
- 400: Bad Request (invalid input data)
- 401: Unauthorized (missing or invalid API key)
- 422: Validation Error (data validation failed)
- 429: Too Many Requests (rate limit exceeded)
- 500: Internal Server Error
- 503: Service Unavailable (model not loaded)