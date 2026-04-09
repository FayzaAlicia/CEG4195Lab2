# Lab 2 - House Segmentation Service

## 4) Run the API locally
```bash
python app.py
```

## Test  API
```bash
curl -X POST http://localhost:5000/predict \
  -F "image=@data/sample_aerial.png"
```

##  Run with Docker
```bash
docker build -t house-segmentation-service .
docker run --env-file .env -p 5000:5000 house-segmentation-service
```

Or:
```bash
docker compose up --build
```

