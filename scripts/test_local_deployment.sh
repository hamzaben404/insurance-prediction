#!/bin/bash
# Test local deployment before pushing to Railway

echo "🧪 Testing local deployment setup..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

echo "1️⃣ Building Docker image..."
docker build -t insurance-api-test .

echo "2️⃣ Running container..."
docker run -d -p 8000:8000 --name insurance-api-local \
    -e GITHUB_REPOSITORY=hamzaben404/insurance-prediction \
    -e MODEL_VERSION=latest \
    insurance-api-test

echo "3️⃣ Waiting for container to start..."
sleep 10

echo "4️⃣ Testing health endpoint..."
if curl -f http://localhost:8000/health; then
    echo "✅ Health check passed!"
else
    echo "❌ Health check failed. Checking logs..."
    docker logs insurance-api-local
fi

echo "5️⃣ Testing prediction endpoint..."
curl -X POST http://localhost:8000/predictions/predict \
    -H "Content-Type: application/json" \
    -d '{
        "gender": "Male",
        "age": 35,
        "has_driving_license": true,
        "region_id": 28,
        "vehicle_age": "1-2 Year",
        "past_accident": "No",
        "annual_premium": 2630,
        "sales_channel_id": 26,
        "days_since_created": 80
    }'

echo -e "\n\n6️⃣ Cleaning up..."
docker stop insurance-api-local
docker rm insurance-api-local

echo "✅ Local deployment test complete!"
