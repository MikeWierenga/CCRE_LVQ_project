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
    
def custom_metric(x, y):
    print(x, y)
    test = np.sqrt(np.sum(x-y)**2)  
    return test


def ccre_distance(x, y):
    print(x[0], y[0])
    # Caluclating CRE
    x= np.array(x).reshape(-1,1)
    y = np.array(y).reshape(-1,1)
    cre_x = cre.CRE(x)
    cre_x_value = cre_x.cre_gaussian_distribution()
    cre_y = cre.CRE(y)
    cre_y_value = cre_y.cre_gaussian_distribution()
    if cre_x_value == cre_y_value:
        return 0
    #calculate expecation value X|Y
    mean_y = np.mean(y)
    sigma_y = np.std(y)

    new_data= np.concatenate((x, y), axis = 1)

    ccre_distance = ccre.CCRE(new_data.T)
    cov_conditional_dist = ccre_distance.cov_conditional_distribution() 
    expect_value_cre_xy = integrate.dblquad(ccre_distance.calculate_expectation_value_xy, -np.inf, np.inf, np.mean(x), np.inf, args=(mean_y, sigma_y, cov_conditional_dist))[0]
    ccre_value = (cre_x_value + (expect_value_cre_xy))/cre_x_value
    if ccre_value <= 0:
        return 1
    else:

        return 1 - ccre_value

