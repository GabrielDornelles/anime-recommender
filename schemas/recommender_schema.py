from pydantic import BaseModel


class RecommenderRequest(BaseModel):
    query: str
    recommendation_size: int
    mode: str