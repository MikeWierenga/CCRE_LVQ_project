import kNN
import numpy as np
from scipy import integrate
import sys
import os
sys.path.append('/home/fcourse1/Desktop/afstudeerstage/code/CCRE_LVQ_project/Calculating_distance_measures/data')
import load_data
sys.path.append('/home/fcourse1/Desktop/afstudeerstage/code/CCRE_LVQ_project/Calculating_distance_measures/CCRE_distance')
import cre 
import ccre
import pandas as pd
from imblearn.over_sampling import SMOTE

class main:
  def __init__(self) -> None:
     pass
  
  def load_data(self):
        # loading the data
        data = load_data.Load_Data()
        feature_vectors = data.get_dataframe('feature_vectors')
        feature_vectors = feature_vectors.set_axis(list(range(1, 36)), axis=1)
        center = data.get_dataframe('center_label')
        center.rename(columns={0: "center_label"}, inplace=True)

        diagnosis = data.get_dataframe('diagnosis_label')
        diagnosis.rename(columns={0: "diagnosis_label"}, inplace=True)

        df = data.combine_dataframes(center, diagnosis, feature_vectors)
        
      
        return df
  
  def ccre_distance(self, x, y):
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
      
  def main(self):
    #load data
  
    data = self.load_data()
    patients_to_remove = [106,109,110,130,154, 156, 159, 23, 25, 27,30,32,34,35,36,40,52, 56, 104,110,111,117,120,134,140,148,159,161,162, 171,179,180,181,182,191,192,196,205,211,215,216, 230,233,236,237,238,239,242,243,245, 247,252,253,254,256,258,262,272,275,277,279,284,285,286,287,289,290,291,292,294,295,296,297,298,299,300,301,303]
    data = data.drop(index=patients_to_remove)
    #oversampling with SMOTE because it does not duplicate rows and is not only looking at outliers like adasyn does. link: https://imbalanced-learn.org/stable/over_sampling.html

    X = data.loc[:, 1:34].values
    y = list(data['center_label'] + data['diagnosis_label'])
    X_resampled, y_resampled = SMOTE().fit_resample(X[:45], y[:45])
  
    #model
    knn_model = kNN.KNN()
    n_neighbors = np.linspace(1,10, num=10)
    weights = ['uniform', 'distance']
    algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
    n_jobs = [8]
    metrics = ['euclidean', self.ccre_distance] 
    knn_model.set_parameters(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm,metric=metrics, n_jobs=n_jobs)
    
    gridsearch = knn_model.gridsearch().fit(X_resampled, y_resampled)
    
    

test = main()
test.main()
