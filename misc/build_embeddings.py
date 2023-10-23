import torch
from transformers import BigBirdTokenizer, BigBirdModel, BertModel, BertTokenizer
import json
import pickle

def synopsis_to_embeddings(synopsis: str):
    # Check for GPU availability and set device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pre-trained BigBird tokenizer and model on GPU
    tokenizer = BigBirdTokenizer.from_pretrained('google/bigbird-roberta-base')
    model = BigBirdModel.from_pretrained('google/bigbird-roberta-base').to(device)


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

def title_to_embeddings(title: str):
       # Check for GPU availability and set device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pre-trained BERT tokenizer and model on GPU
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased').to(device)

    tokens = tokenizer.encode(title, add_special_tokens=True, truncation=True, max_length=512)
    # Convert the tokens to PyTorch tensors on GPU
    input_ids = torch.tensor([tokens]).to(device)

    # Get the BERT model's output embeddings
    with torch.no_grad():
        outputs = model(input_ids)
        embeddings = outputs.last_hidden_state

    # Extract the embeddings for each token in the synopsis
    synopsis_embeddings = embeddings.squeeze(0)

    # Average the embeddings to get a single embedding for the entire synopsis
    avg_synopsis_embedding = torch.mean(synopsis_embeddings, dim=0)
    avg_synopsis_embedding = avg_synopsis_embedding.cpu().numpy()
    return avg_synopsis_embedding

# Open the JSON file
with open('dataset.json') as file:
    # Load the JSON data into a list
    dataset = json.load(file)

anime_embeddings_dict = {}
name_embeddings_dict = {}
no_synopsis_anime = []
for idx, item in enumerate(dataset):
    anime_name = list(item.keys())[0]
  
    synopsis = item[anime_name]["synopsis"]
    title = anime_name
    if str(synopsis).lower() == "none": synopsis = "None"
    if synopsis == "None":
        no_synopsis_anime.append(anime_name)
        
    print(f"Running: {idx}/{len(dataset)} => {anime_name}.")
    synopsis_embeddings = synopsis_to_embeddings(synopsis)
    title_embeddings = title_to_embeddings(title)
    anime_embeddings_dict[anime_name] = {
        "embeddings": synopsis_embeddings,
        "data": item
    }
    name_embeddings_dict[anime_name]: title_embeddings

print(f"anime_embeddings_dict length: {len(anime_embeddings_dict)}")
print(f"name_embeddings_dict length: {len(name_embeddings_dict)}")


with open('synopsis_embeddings_dict.pickle', 'wb') as f:
    pickle.dump(anime_embeddings_dict, f)

with open('name_embeddings_dict.pickle', 'wb') as f:
    pickle.dump(anime_embeddings_dict, f)