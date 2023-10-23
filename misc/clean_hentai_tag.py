import json
import pickle

# Open the JSON file
with open('dataset.json') as file:
    # Load the JSON data into a list
    dataset = json.load(file)

with open('data/anime_name_list.pickle', 'rb') as f:
    anime_list = pickle.load(f)

print(f"Dataset length before hentai remove: {len(dataset)}")
print(f"Anime List length before hentai remove: {len(anime_list)}")

hentais = []
idxs_to_remove = []
for idx, item in enumerate(dataset):
    anime_name = list(item.keys())[0]
    print(f"Running {idx} - {anime_name}")
    genres = item[anime_name]["genres"]
    genres = [item["name"].lower() for item in genres]
    if "hentai" in genres:
        # del dataset_cpy[idx]
        # del anime_list_cpy[idx]
        idxs_to_remove.append(idx)
        hentais.append(anime_name)

print(f"There is {len(hentais)} Hentais. ")

new_dataset = []
new_anime_list = []
for idx, item in enumerate(dataset):
    if idx not in idxs_to_remove:
        new_anime_list.append(anime_list[idx])
        new_dataset.append(dataset[idx])
    

print(f"Dataset length after hentai remove: {len(new_dataset)}")
print(f"Anime List length after hentai remove: {len(new_anime_list)}")


with open('new_anime_names.pickle', 'wb') as f:
    pickle.dump(new_anime_list, f)

with open('new_dataset.json', 'w') as file:
    # Write the list into the file as JSON
    json.dump(new_dataset, file)


