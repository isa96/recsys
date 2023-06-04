import numpy as np
import pandas as pd
from surprise import Reader, Dataset, SVD

class CollabRecSys:
    def __init__(self, data):
        self.df = pd.read_csv(data)
        self.all_dest = self.df['nama'].unique()
        self.model = None

    def fit(self):
        data = Dataset.load_from_df(self.df, Reader())
        trainset = data.build_full_trainset()
        self.model = SVD()
        self.model.fit(trainset)

    def recommend(self, user_id, topk=10):
        visited = self.df[self.df.user_id == user_id].nama
        not_visited = [dest for dest in self.all_dest if dest not in visited.values]
        score = [self.model.predict(user_id, dest).est for dest in not_visited]

        result = pd.DataFrame({'Destination': not_visited, 'Rating_Prediction':score})
        result.sort_values("Rating_Prediction", ascending=False, inplace=True)
        return result.head(topk)
