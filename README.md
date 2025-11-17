# Sentiment Analysis API Backend

This repository contains the backend service for a simple NLP sentiment classifier. It uses a REST API endpoint that analyzes the sentiment of a given text using a pre-trained Hugging Face model.

## Tech Stack

*   Python
*   FastAPI
*   Transformers (Hugging Face)
*   PyTorch
*   Uvicorn (ASGI server)

## Endpoints

*   `GET /`: Returns a simple status message.
*   `POST /predict`:
    *   **Request Body:** `{"text": "string"}`
    *   **Response:** `{"label": "string", "score": "float"}`

## Deployment

This backend is designed to be deployed on [Render](https://render.com/). It uses `requirements.txt` for dependencies and the start command `uvicorn main:app --host 0.0.0.0 --port $PORT`.

## Local Setup (Optional)

1.  Create a virtual environment: `python -m venv venv`
2.  Activate it: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
3.  Install dependencies: `pip install -r requirements.txt`
4.  Run the server: `uvicorn main:app --reload`
5.  The API will be available at `http://127.0.0.1:8000`.
