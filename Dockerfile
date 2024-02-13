# Use the official Python image from Docker Hub
FROM python:3.10

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

COPY requirements.txt requirements.txt
COPY app.py app.py
COPY routers routers
COPY misc misc
COPY services services
COPY data data
COPY schemas schemas
COPY static static

# Install dependencies
# --no-cache-dir
RUN pip install -r requirements.txt 

# Expose port 80 for the FastAPI app
EXPOSE 8000

# Command to run the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
