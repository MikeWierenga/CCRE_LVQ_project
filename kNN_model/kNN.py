from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import scipy.stats as ss
from scipy import integrate
import sys
sys.path.append('/home/fcourse1/Desktop/afstudeerstage/code/CCRE_LVQ_project/Calculating_distance_measures/CCRE_distance')
import cre 
import ccre
class KNN:
    def __init__(self):
        self.neighbours = 5
        self.weights = 'uniform'
        self.algorithm = 'auto'
        self.leaf_size = 30
        self.p = 2
        self.metric = 'minkowski'
        self.n_jobs = -1
        self.model = self.create_model()
     
    def create_model(self):
        model = KNeighborsClassifier()
        return model
    
    def set_parameters(self, n_neighbors =5, weights='uniform', algorithm='auto', leaf_size=30, p =2, metric='minkowski', n_jobs=-1):
         self.neighbours = n_neighbors
         self.weights = weights
         self.algorithm = algorithm
         self.leaf_size = leaf_size
         self.p = p
         self.metric = metric
         self.n_jobs = n_jobs

    def gridsearch(self):
        params = [self.neighbours, self.weights, self.algorithm, self.leaf_size, self.p, self.metric, self.n_jobs]
        model = GridSearchCV(self.model, param_grid=params, n_jobs=-1)
        return model
    
    def make_predict(self, model, new_entry):
        return model.predict(new_entry)
    
    def calc_score(self, model, X, y):
        return model.score(X, y)

