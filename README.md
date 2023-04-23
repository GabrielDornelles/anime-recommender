# anime_recommender
Simple synopsis based recommender. Embeddings are projected with Bigbird-ROBERTA, kNN and SVM supported for recommendations.

Built with FastAPI. Frontend page built with ChatGPT (I know nothing about html/css/js, I just re-arrange what it writes).

## Run

```sh
uvicorn app:app # or
gunicorn 0.0.0.0:8000 --daemon app:app # deploy it somewhere
```

## Use

Access ```http://127.0.0.1:8000```, frontend will be displayed on ```/``` (no route, root).

