import pickle
import numpy as np
from sklearn import svm
import editdistance


class AnimeRecommender:

    def __init__(self) -> None:
        with open("data/embeddings_full_table.pickle", 'rb') as f:
            self.embeddings_table = pickle.load(f)
        #embeddings = np.array(list(self.embeddings_table.values()))
        embeddings = np.array([v["embeddings"] for v in self.embeddings_table.values()])
        embeddings = embeddings / np.sqrt((embeddings**2).sum(1, keepdims=True))
        self.embeddings = embeddings
        with open('data/anime_names.pickle', 'rb') as f:
            self.anime_list = pickle.load(f)
    
    def __call__(self, query: str, recommendation_size: int, mode: str = "svm") -> list:
        return self.get_recommendations(query, recommendation_size, mode)
  
    @staticmethod
    def get_most_similar_string(query, strings):
        min_distance = float('inf')
        most_similar_string = None
        for string in strings:
            distance = editdistance.eval(query, string)
            if distance < min_distance:
                min_distance = distance
                most_similar_string = string
        return most_similar_string
    
    def get_recommendations(self, query: str, 
            recommendation_size: int, 
            mode: str = "svm"
        ) -> list:
        query = self.get_most_similar_string(query, self.anime_list)
        query = self.embeddings_table[query]["embeddings"]
        query = query / np.sqrt((query**2).sum())

        if mode.lower() == "svm":
            x = self.embeddings
            y = np.zeros(len(self.embeddings_table))
            # Find the idx of the query using cosine similarity, it should be 1 with itself
            #similarities = cosine_similarity(x, query.reshape(1, -1)).ravel()
            #idx = np.argmax(similarities)
            query_idx = np.where(np.all(self.embeddings == query, axis=1))[0][0]
            y[query_idx] = 1
            clf = svm.LinearSVC(class_weight='balanced', verbose=False, max_iter=10000, tol=1e-6, C=1.0)
            clf.fit(x, y) # train
            similarities = clf.decision_function(x)
            sorted_ix = np.argsort(-similarities)
        if mode.lower() == "knn":
            similarities = self.embeddings.dot(query)
            sorted_ix = np.argsort(-similarities)
        
        recommendations = []
        for idx, k in enumerate(sorted_ix[:recommendation_size]):
            anime_name = list(self.embeddings_table.keys())[k]
            image = self.embeddings_table[anime_name]["image"]
            genres = self.embeddings_table[anime_name]["genres"]
            buffer = {
                "image": image,
                "name": anime_name,
                "genres": genres
            }
            recommendations.append(buffer)
        return recommendations
