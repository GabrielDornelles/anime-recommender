from fastapi_utils.cbv import cbv
from fastapi_utils.inferring_router import InferringRouter
#from fastapi import Request
from schemas.recommender_schema import RecommenderRequest
from fastapi.responses import JSONResponse

from services.svm_recommender import AnimeRecommender

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

router = InferringRouter()
recommender = AnimeRecommender()

router.mount("/static", StaticFiles(directory='static'), name="static")

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

    @router.get("/home")
    def read_file(path: str):
        return FileResponse("static/index.html")
        
    @router.post(
    '/search',
    summary="",
    description="",
    response_description="")
    def search(self, data: RecommenderRequest):
        query = data.query
        recommendation_size = data.recommendation_size
        mode = data.mode.lower()
        recommendations = recommender(query, recommendation_size, mode=mode)
        return JSONResponse(status_code=200,content=recommendations)