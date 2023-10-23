import requests
import json


anime_data = {}

for i in range(1,1021): # update this as animes get aired
    try:
        result = requests.get(f"https://api.jikan.moe/v4/anime?page={i}")
        print(f"requesting page: {i}")
        for item in result.json()["data"]:
            buffer = {
                item["title"]: item
            }
            with open("/home/gabriel-dornelles/Documents/gabriel/anime_recommender/dataset.json", "a") as outfile:
                json.dump(buffer , outfile, indent=4)
                outfile.write(",\n")
    except:
        print(f"Request failed for page {i}, trying again")
        result = requests.get(f"https://api.jikan.moe/v4/anime?page={i}")
        print(f"requesting page: {i}")
        for item in result.json()["data"]:
            buffer = {
                item["title"]: item
            }
            with open("/home/gabriel-dornelles/Documents/gabriel/anime_recommender/dataset.json", "a") as outfile:
                json.dump(buffer , outfile, indent=4)
                outfile.write(",\n")
        
