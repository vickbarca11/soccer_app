FROM python:3.11
WORKDIR /app

COPY data-backend/requirements.txt .

RUN pip install -r requirements.txt

COPY data-backend/ .

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
