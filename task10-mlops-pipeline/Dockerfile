# FastAPI service image for the Task 10 MLOps pipeline.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /srv

# Install Python deps first so the (large) sklearn layer is cached between edits.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Application code + shipped Task 9 artifacts.
COPY main.py schemas.py ./
COPY models/ models/

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=8s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]