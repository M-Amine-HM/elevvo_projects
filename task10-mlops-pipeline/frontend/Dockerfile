# Gradio frontend image for the Task 10 MLOps pipeline.
# Deliberately does NOT install scikit-learn/joblib/xgboost — the UI is a thin
# HTTP client that talks to the FastAPI service, which owns the model.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /srv

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ./

EXPOSE 7860

CMD ["python", "app.py"]