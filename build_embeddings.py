import torch
from transformers import BigBirdTokenizer, BigBirdModel

# Check for GPU availability and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained BigBird tokenizer and model on GPU
tokenizer = BigBirdTokenizer.from_pretrained('google/bigbird-roberta-base')
model = BigBirdModel.from_pretrained('google/bigbird-roberta-base').to(device)

def synopsis_to_embeddings(synopsis: str):
    tokens = tokenizer.encode(synopsis, add_special_tokens=True, truncation=True, max_length=512)
    # Convert the tokens to PyTorch tensors on GPU
    input_ids = torch.tensor([tokens]).to(device)

    # Get the BigBird model's output embeddings
    with torch.no_grad():
        outputs = model(input_ids)
        embeddings = outputs.last_hidden_state

    # Extract the embeddings for each token in the synopsis
    synopsis_embeddings = embeddings.squeeze(0)

    # Average the embeddings to get a single embedding for the entire synopsis
    avg_synopsis_embedding = torch.mean(synopsis_embeddings, dim=0)
    avg_synopsis_embedding = avg_synopsis_embedding.cpu().numpy()
    return avg_synopsis_embedding

import json

# Open the JSON file
with open('new_dataset.json') as file:
    # Load the JSON data into a list
    dataset = json.load(file)

anime_embeddings_dict = {}
no_synopsis_anime = []
for idx, item in enumerate(dataset):
    anime_name = list(item.keys())[0]
  
    synopsis = item[anime_name]["synopsis"]
    if str(synopsis).lower() == "none": synopsis = "None"
    if synopsis == "None":
        no_synopsis_anime.append(anime_name)
        
    print(f"Running: {idx}.     Synopsis: {synopsis}")
    #print(synopsis)
    embeddings = synopsis_to_embeddings(synopsis)
    anime_embeddings_dict[anime_name] = {
        "embeddings": embeddings,
        "data": item
    }

print(f"anime_embeddings_dict length: {len(anime_embeddings_dict)}")

import pickle

with open('new_anime_embeddings_dict.pickle', 'wb') as f:
    pickle.dump(anime_embeddings_dict, f)