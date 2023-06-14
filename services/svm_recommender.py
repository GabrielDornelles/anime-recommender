import pickle
import numpy as np
from sklearn import svm
import editdistance
from pymongo import MongoClient
#import torch
#from transformers import BertTokenizer, BertModel

client = MongoClient("mongodb://localhost:27017/")
db = client["animelist"]
collection = db["gogaido_beta"]

class AnimeRecommender:

    def __init__(self) -> None:
        self._init_synopsis_embeddings()
        #self._init_anime_name_embeddings()
        #self._init_bert()
    
    # def _init_bert(self):
    #     self.device = torch.device("cpu")
    #     self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #     self.bert = BertModel.from_pretrained('bert-base-uncased').to(self.device)
  
    def _init_synopsis_embeddings(self):
        with open("data/anime_embeddings_dict.pickle", 'rb') as f:
            self.embeddings_table = pickle.load(f)
        embeddings = np.array([v["embeddings"] for v in self.embeddings_table.values()])
        embeddings = embeddings / np.sqrt((embeddings**2).sum(1, keepdims=True))
        self.embeddings = embeddings
        del self.embeddings_table
        
        with open('data/anime_names.pickle', 'rb') as f:
            self.anime_list = pickle.load(f)
    
    # def _init_anime_name_embeddings(self):
    #     with open('data/anime_name_embeddings.pickle', 'rb') as f:
    #         anime_name_embeddings = pickle.load(f)
    #     embeddings = np.array([v for v in anime_name_embeddings.values()])
    #     embeddings = embeddings / np.sqrt((embeddings**2).sum(1, keepdims=True))
    #     self.anime_name_embeddings = embeddings
    
    def __call__(self, query: str, recommendation_size: int, mode: str = "svm") -> list:
        return self.get_recommendations(query, recommendation_size, mode)

    # def bert_inference(self, query: str) -> np.array:
    #     tokens = self.tokenizer.encode(query, add_special_tokens=True)
    #     input_ids = torch.tensor([tokens]).to(self.device)
    #     with torch.no_grad():
    #         outputs = self.bert(input_ids)
    #         embeddings = outputs.last_hidden_state
    #     synopsis_embeddings = embeddings.squeeze(0)
    #     avg_synopsis_embedding = torch.mean(synopsis_embeddings, dim=0)
    #     avg_synopsis_embedding = avg_synopsis_embedding.cpu().numpy()
    #     return avg_synopsis_embedding

    def get_most_similar_string(self, query):
        min_distance = float('inf')
        most_similar_string = None
        for string in self.anime_list:
            distance = editdistance.eval(query, string)
            if distance < min_distance:
                min_distance = distance
                most_similar_string = string
        return most_similar_string
    
    # def get_most_similar_string(self, query):
    #     query_embeddings = self.bert_inference(query)
    #     query_embeddings = query_embeddings / np.sqrt((query_embeddings**2).sum())
    #     similarities = self.anime_name_embeddings.dot(query_embeddings)
    #     sorted_ix = np.argsort(-similarities)
    #     return self.anime_list[sorted_ix[0]]
    
    def get_recommendations(self, query: str, 
            recommendation_size: int, 
            mode: str = "svm",
        ) -> list:

        query_anime_name = self.get_most_similar_string(query)
        query = self.embeddings[self.anime_list.index(query_anime_name)]
        query = query / np.sqrt((query**2).sum())

        if mode.lower() == "svm":
            x = self.embeddings
            y = np.zeros(x.shape[0])
            query_idx = self.anime_list.index(query_anime_name)#[0][0]#np.where(np.all(self.embeddings == query, axis=1))#[0][0]
            y[query_idx] = 1
            clf = svm.LinearSVC(class_weight='balanced', verbose=False, max_iter=10000, tol=1e-6, C=1.0)
            clf.fit(x, y) # train
            similarities = clf.decision_function(x)
            sorted_ix = np.argsort(-similarities)
        if mode.lower() == "knn":
            similarities = self.embeddings.dot(query)
            sorted_ix = np.argsort(-similarities)
        
        recommendations = []
    
        for idx in (sorted_ix):

            anime_name = self.anime_list[idx]
            if '.' in anime_name:
                query_name = anime_name.replace(".", "_")
            else:
                query_name = anime_name
            db_query = {query_name: {"$exists": True}}
            results = collection.find(db_query)
            data =  list([result for result in results][0].values())[1]
            buffer = {
                "name": anime_name,
                "data": data
            }
            recommendations.append(buffer)
            if len(recommendations) == recommendation_size + 1: 
                break
        return recommendations

        # Clean Hentai Tag
        # hentai_flag = False
        # for idx, k in enumerate(sorted_ix):
        #     anime_name = list(self.embeddings_table.keys())[k]
        #     data = self.embeddings_table[anime_name]["data"]


        #     genres = data[anime_name]["genres"]
        #     for genre in genres:
        #         if genre["name"].lower() == "hentai":
        #             hentai_flag = True
                
        #     if hentai_flag:
        #         hentai_flag = False
        #         continue
            
        #     buffer = {
        #         "name": anime_name,
        #         "data": data[anime_name]
        #     }
        #     recommendations.append(buffer)
        #     if len(recommendations) == recommendation_size + 1: 
        #         break
        # return recommendations

    


