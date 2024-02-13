import pickle
import numpy as np
from sklearn import svm
import editdistance
from pymongo import MongoClient
import torch
from transformers import BertTokenizer, BertModel

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  # Import 3D plotting

client = MongoClient("mongodb:27017")
db = client["anime_recommender_db"]
collection = db["gogaido_beta"]

# TODO: break this all into smaller services

class AnimeRecommender:

    def __init__(self) -> None:
        self._init_synopsis_embeddings()
        self._init_anime_name_embeddings()
        self._init_bert()
    
    def _init_bert(self):
        self.device = torch.device("cpu")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased').to(self.device)
  
    def _init_synopsis_embeddings(self):
        with open("data/anime_embeddings_dict.pickle", 'rb') as f:
            self.embeddings_table = pickle.load(f)
        embeddings = np.array([v["embeddings"] for v in self.embeddings_table.values()])
        embeddings = embeddings / np.sqrt((embeddings**2).sum(1, keepdims=True))
        self.embeddings = embeddings
        del self.embeddings_table
        
        with open('data/anime_names.pickle', 'rb') as f:
            self.anime_list = pickle.load(f)
    
    def _init_anime_name_embeddings(self):
        with open('data/anime_name_embeddings.pickle', 'rb') as f:
            anime_name_embeddings = pickle.load(f)
        embeddings = np.array([v for v in anime_name_embeddings.values()])
        embeddings = embeddings / np.sqrt((embeddings**2).sum(1, keepdims=True))
        self.anime_name_embeddings = embeddings
    
    def __call__(self, query: str, recommendation_size: int, mode: str = "svm") -> list:
        return self.get_recommendations(query, recommendation_size, mode)

    def bert_inference(self, query: str) -> np.array:
        tokens = self.tokenizer.encode(query, add_special_tokens=True)
        input_ids = torch.tensor([tokens]).to(self.device)
        with torch.no_grad():
            outputs = self.bert(input_ids)
            embeddings = outputs.last_hidden_state
        synopsis_embeddings = embeddings.squeeze(0)
        avg_synopsis_embedding = torch.mean(synopsis_embeddings, dim=0)
        avg_synopsis_embedding = avg_synopsis_embedding.cpu().numpy()
        return avg_synopsis_embedding

    # def get_most_similar_string(self, query): # alternate version if BERT cant be used
    #     min_distance = float('inf')
    #     most_similar_string = None
    #     for string in self.anime_list:
    #         distance = editdistance.eval(query, string)
    #         if distance < min_distance:
    #             min_distance = distance
    #             most_similar_string = string
    #     return most_similar_string
    
    def get_most_similar_string(self, query):
        query_embeddings = self.bert_inference(query)
        query_embeddings = query_embeddings / np.sqrt((query_embeddings**2).sum())
        similarities = self.anime_name_embeddings.dot(query_embeddings)
        sorted_ix = np.argsort(-similarities)
        return self.anime_list[sorted_ix[0]]
    
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
            #clf = svm.LinearSVC(class_weight='balanced', verbose=False, max_iter=10000, tol=1e-6, C=1.0)
            clf = svm.SVC(class_weight="balanced", verbose=False, max_iter=10000, tol=1e-6, C=1.0)
            clf.fit(x, y) # train
            similarities = clf.decision_function(x)
            sorted_ix = np.argsort(-similarities)
            
            # pca = PCA(n_components=3)
            # x_pca = pca.fit_transform(x)

            # top_30_indices = sorted_ix[1:30]
            # top_1_index = sorted_ix[:1]
            # x_top_30 = x_pca[top_30_indices]
            # y_top_30 = y[top_30_indices]

            # x_top_1 = x_pca[top_1_index]

            # # Select the last 100 data points
            # last_100_indices = sorted_ix[-100:]
            # x_last_100 = x_pca[last_100_indices]
            # y_last_100 = y[last_100_indices]

            # ## Create a PCA object and fit it to your data
            # pca = PCA(n_components=3)
            # x_pca = pca.fit_transform(x)

            # # Create a scatter plot of the last 100 data points in red
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')

            # ax.scatter(x_last_100[:, 0], x_last_100[:, 1], x_last_100[:, 2], c='r', label='Top 100 - Least similar')

            # # Plot the top 30 most similar data points in blue
            # ax.scatter(x_top_30[:, 0], x_top_30[:, 1], x_top_30[:, 2], c='b', label='Top 30 - Most similar')

            # # Plot the point representing the trained model in green
            # #model_point = pca.transform(clf.coef_)
            # anime_name = self.anime_list[sorted_ix[0]]
            # ax.scatter(x_top_1[:,0], x_top_1[:,1], x_top_1[:,2], s=100, marker='o', c='g', label=f'Trained point - Title: {anime_name}')

            # Get the coefficients of the decision boundary (hyperplane)
            # coeff = clf.coef_[0]
            # intercept = clf.intercept_[0] #+ 1.2

            # # Create a meshgrid for the decision boundary
            # x_min, x_max = x_pca[:, 0].min() - 1, x_pca[:, 0].max() + 1
            # y_min, y_max = x_pca[:, 1].min() - 1, x_pca[:, 1].max() + 1
            # xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
            # zz = -(coeff[0] * xx + coeff[1] * yy + intercept) / coeff[2]
    
            # # Plot the decision boundary as a 3D surface
            # ax.plot_surface(xx, yy, zz, alpha=0.5, rstride=100, cstride=100, color='k', label='Decision Boundary')

            # ax.set_xlabel('PCA Component 1')
            # ax.set_ylabel('PCA Component 2')
            # ax.set_zlabel('PCA Component 3')
            # plt.legend()
            # plt.title('PCA on Big Bird embeddings and SVC hyperplane')
            # plt.show()
                    
           
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
            try:
                data =  list([result for result in results][0].values())
            except IndexError: # this shouldn't happen, I just messed by db and didn't repopulate it, some items were deleted
                continue
        
            data = data[1]
            buffer = {
                "name": anime_name,
                "data": data
            }
            
            recommendations.append(buffer)
            if len(recommendations) == recommendation_size + 1: 
                break
        return recommendations
