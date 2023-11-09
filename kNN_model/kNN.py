from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import scipy.stats as ss
from scipy import integrate
import sys
sys.path.append('/home/fcourse1/Desktop/afstudeerstage/code/CCRE_LVQ_project/Calculating_distance_measures/CCRE_distance')
import cre 
import ccre
class KNN:
    def __init__(self, neighbours, metric):
        self.neighbours = neighbours
        self.metric = metric
        self.x = []
        self.y = []
    
    def create_model(self):
        model = KNeighborsClassifier(n_neighbors=self.neighbours, metric=self.metric, n_jobs=-1)
        return model
    
    def fit_model(self, x, y):
        self.x = x
        self.y = y

    def make_predict(self, model, new_entry):
        return model.predict(new_entry)
    
    def calc_score(self, model, X, y):
        return model.score(X, y)

