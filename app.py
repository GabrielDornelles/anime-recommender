import os
import uvicorn
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI
from routers.routes import router
from fastapi.middleware.cors import CORSMiddleware

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://gogaido.vercel.app/"],  # Set this to restrict the origins if desired
    allow_credentials=True,
    allow_methods=["*"],  # Set the allowed HTTP methods
    allow_headers=["*"],  # Set the allowed headers
)
app.include_router(router)

if __name__ == '__main__':
    uvicorn.run(app=app, port=int(os.getenv('PORT', 8000)))
