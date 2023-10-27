from sklearn.neighbors import KNeighborsClassifier
import numpy as np
class KNN:
    def __init__(self, neighbours, metric):
        self.neighbours = neighbours
        self.metric = metric
        self.x = []
        self.y = []
    
    def create_model(self):
        model = KNeighborsClassifier(n_neighbors=self.neighbours, metric=self.metric)
        return model
    
    def fit_model(self, x, y):
        self.x = x
        self.y = y

    def make_predict(self, model, new_entry):
        return model.predict(new_entry)
    
    def calc_score(self, model, X, y):
        return model.score(X, y)
    
def custom_metric(x, y):
    test = np.sqrt(np.sum(x-y)**2)  
    return test

X = [[0], [1], [2], [3]]
y = [0, 0, 1, 1]
knn_test = KNeighborsClassifier(n_neighbors=3, metric=custom_metric)
knn_test.fit(X, y)
print(knn_test.predict([[1.1]]))
print(knn_test.predict_proba([[0.9]]))

