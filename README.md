# anime-recommender
A Simple synopsis based anime recommender. Embeddings are projected with BigBird-ROBERTA.

You can see its embeddings are really good since the model not only reduce the dimensionality of the data (synopsis) but it also preserves semantic relationships. See how it recommends a lot of mecha genre anime when asked to recommend animes like Evangelion (remember it has no knowledge of tags or titles, it only reads synopsis):

![image](/docs/images/evangelion_search.png)

We can also see that the recommended embeddings trained by an SVC are mostly close to each other when plotted in a 3d plane with PCA:
![image](/docs/images/pca_on_embeddings.jpg)

Although the hyperplane itself is not visually intuitive, remember embeddings are 768d not 3d (and you can see on the left plot a blue dot inside the red ones, to remember that not only the 3 dimensions with most variance are being used when recommending). A lightweight recommendation is also possible using a kNN instead of the SVM approach.

## Search name matching
We use a BERT model to perform name matching between the searched anime and the animes we have in the database.

## The simplest Frontend

This is a very simple frontend written by ChatGPT just to interact better with the API.

![image](https://user-images.githubusercontent.com/56324869/233866039-1a8fc973-dc18-4eda-96c7-8ad50523f70a.png)

## We have a dedicated server and frontend!

We are using a free AWS machine to host part of this backend (until february 2024, when Amazon will start to charge for it 😔). Unfortunately, the free VM is very limited ([t2.micro](https://instances.vantage.sh/aws/ec2/t2.micro)), and it can't handle model inference, so the name matching BERT is disabled and a very simple algorithm is being used instead.

We do also have a dedicated frontend! See: https://gogaido.vercel.app/

![image](docs/images/gogaido_example.png)
## How to reproduce it

First, you should download [those files](https://drive.google.com/file/d/1-ddrmsloUfGAzJ8Ti4VBOhcnnT30Z4t0/view?usp=share_link) and place them under a directory named ```/data```, those are the already calculated embbedings for ~23k animes. 

Additionaly, you can [download this dataset](https://drive.google.com/file/d/1ZvRBJ9TvmHdbu-KZAxEwIkrE1gZq8Cog/view?usp=share_link) to populate the database, or run the embeddings yourself with a different model.

Then, create a MongoDB instance and populate it with:

```sh
# Install mongo: see https://www.mongodb.com/docs/mongodb-shell/install/
sudo service mongod start 
python3 misc/pickle_to_mongo.py
```

Now you should have a database that the API will use to retrieve data:

![image](/docs/images/mongo_example.png)
Then you can run the app with:

```sh
uvicorn app:app # or
gunicorn 0.0.0.0:8000 --daemon app:app -k uvicorn.workers.UvicornWorker # deploy it somewhere
```

Access ```http://127.0.0.1:8000/home```, frontend will be displayed on ```/home```.

