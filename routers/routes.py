from fastapi_utils.cbv import cbv
from fastapi_utils.inferring_router import InferringRouter
from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from services.svm_recommender import AnimeRecommender

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse


router = InferringRouter()
recommender = AnimeRecommender()

router.mount("/static", StaticFiles(directory='static'), name="static")

class Request(BaseModel):
    query: str
    recommendation_size: int
    mode: str

@cbv(router)
class RecommenderRoutes:

    @router.get(
    '/status',
    tags=["Status"],
    summary="Status da API",
    description="Rota para validação",
    response_description="Retorna OK se a API estiver up")
    def status(self):
        return {"status": "OK"}

    @router.get("/")
    def read_file(path: str):
        return FileResponse("static/index.html")
        
    @router.post(
    '/search',
    summary="",
    description="",
    response_description="")
    def search(self, data: Request):
        query = data.query
        recommendation_size = data.recommendation_size
        mode = data.mode.lower()
        # TODO: apply something to map query to right name in mapped anime list
        print(mode)
        recommendations = recommender(query, recommendation_size, mode=mode)
        return JSONResponse(status_code=200,content=recommendations)