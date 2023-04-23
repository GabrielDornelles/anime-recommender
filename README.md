# anime_recommender
Simple synopsis based recommender. Embeddings are projected with Bigbird-ROBERTA, kNN and SVM supported for recommendations.

Built with FastAPI. Frontend page built with ChatGPT (I know nothing about html/css/js, I just re-arrange what it writes).

## How it looks like

![image](https://user-images.githubusercontent.com/56324869/233866039-1a8fc973-dc18-4eda-96c7-8ad50523f70a.png)


## Run

```sh
uvicorn app:app # or
gunicorn 0.0.0.0:8000 --daemon app:app # deploy it somewhere
```

## Use

Access ```http://127.0.0.1:8000```, frontend will be displayed on ```/``` (no route, root).

