import os
import uvicorn
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI
from routers.routes import router

load_dotenv(find_dotenv())
ENVIRONMENT_NAME = os.getenv("ENVIRONMENT_NAME", None)


if (ENVIRONMENT_NAME is not None) and (ENVIRONMENT_NAME.startswith("production")):
    # Disable swagger and redocs on production enviroment.
    app = FastAPI(docs_url=None, redoc_url=None)
else:
    app = FastAPI(
        title="SVM Anime Recommender",
        description="",
        version="0.0.1")

app.include_router(router)

if __name__ == '__main__':
    uvicorn.run(app=app, port=int(os.getenv('PORT', 8000)))