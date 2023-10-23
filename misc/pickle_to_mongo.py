from pymongo import MongoClient
import pickle

client = MongoClient("mongodb://localhost:27017/")
db = client["animelist"]
collection = db["gogaido_beta"]

# Given that you have an embeddings dict (download it or generate it)
with open("data/anime_embeddings_table.pickle", "rb") as f:
    embeddings_table = pickle.load(f)

for k,v in embeddings_table.items():
    print(f"Writing {k} to MongoDB")
    data = v["data"][k]
    if '.' in k:
        k = k.replace(".", "_") # Mongo uses '.' as part of its syntax, so this avoids naming issues
    buffer = {
        k: data
    }
    collection.insert_one(buffer)