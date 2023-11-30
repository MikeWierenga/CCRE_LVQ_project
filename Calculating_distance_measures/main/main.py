
import multiprocessing
import sys
import os
import csv
import multiprocessing as mp 
from multiprocessing import Lock
from functools import partial
sys.path.append('data')


import load_data
sys.path.append('euclidean_distance')
import euclidean_distance
import numpy as np
sys.path.append('CCRE_distance')
import cre 
import ccre
sys.path.append('neighbours')
import neighbours
sys.path.append('write_to_csv')
import csv_file
import pandas as pd
from sklearn import preprocessing
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from scipy.stats import norm
from imblearn.over_sampling import SMOTE, ADASYN
from collections import Counter
import random
class Main():

    def __init__(self, hospital, diagnosis):
        self.hospital = hospital
        self.diagnosis = diagnosis
        self.created_cre_file = False
        self.created_euclidean_file = False
        
    def load_data(self):
        # loading the data
        data = load_data.Load_Data()
        feature_vectors = data.get_dataframe('feature_vectors')
        feature_vectors = feature_vectors.set_axis(list(range(35)), axis=1)
        center = data.get_dataframe('center_label')
        center.rename(columns={0: "center_label"}, inplace=True)

        diagnosis = data.get_dataframe('diagnosis_label')
        diagnosis.rename(columns={0: "diagnosis_label"}, inplace=True)

        df = data.combine_dataframes(center, diagnosis, feature_vectors)
        df = df[(df['center_label'].isin(self.hospital)) & (df['diagnosis_label'].isin(self.diagnosis))]
        patients_to_remove = [23, 25, 27,30,32,34,35,36,40,52, 56, 104,110,111,117,120,134,140,148,159,161,162, 171,179,180,181,182,191,192,196,205,211,215,216, 230,233,236,237,238,239,242,243,245, 247,252,253,254,256,258,262,272,275,277,279,284,285,286,287,289,290,291,292,294,295,296,297,298,299,300,301,303]
        df = df.drop(patients_to_remove)
        df = df.reset_index(drop=True)
        #splitting the data
        df['centre_diagnosis'] = df['center_label'] + '_' + df['diagnosis_label']
        test_ids = []
        for key, value in Counter(df['centre_diagnosis']).items():
            for id in df.index[df['centre_diagnosis'] == key].tolist()[:round(value/100*20)]:
                test_ids.append(id)
       
        testdf =df.loc[df.index[test_ids]]
        traindf = df.loc[~df.index.isin(test_ids)]
     
        testdf['testset'] = True
        X = traindf.loc[:, 0:34].values
     

        
        y = traindf['centre_diagnosis']
        adasyn = SMOTE(random_state=42)
        X_new, y_new = adasyn.fit_resample(X, y)
        
        
            


       
        # X_new, y_new = sm.fit_resample(X, y)
        new_data = np.concatenate((np.array(X_new),np.array(y_new).reshape(-1,1)), axis=1)
        new_df = pd.DataFrame(new_data)
        
        center_label = []
        diagnosis_label = []
        for i in np.array(new_df[35]):
            center_label.append(i.split('_')[0])
            diagnosis_label.append(i.split('_')[1])
        new_df['center_label'] = center_label
        new_df['diagnosis_label'] = diagnosis_label
        new_df.rename(columns={35: "centre_diagnosis"}, inplace=True)
        
        testdf.index = np.arange(len(new_df), len(testdf) + len(new_df))
      
        new_df = pd.concat([new_df, testdf])
        
        testdf.to_csv('test.csv')
        new_df.to_csv('adasyn_df.csv')
        return new_df
    
    def get_neighbours(self):
        connection = neighbours.neighbours(self.load_data())
        return connection.connect_neighbours()
    
    def main(self, lock, df, connections):
        
        
        # create csv file
        with lock:
            if not os.path.isfile(f"CCRE_distances_adasyn.csv"):
            
                    ccre_file = csv_file.CSV(f"CCRE_distances_adasyn.csv")
                    header = ["id_x", "id_y", "hospital", "diagnosis", "cre(x)", "cre(y)", "E[cre(X|Y)]", "ccre(X|Y)"]
                    
                    ccre_file.write_to_csv_file(header)
            
            else:
            
                self.created_cre_file = True
        
            if not os.path.isfile(f"euclidean_distances_adasyn.csv"):
                
                    euclidean_file = csv_file.CSV(f"euclidean_distances_adasyn.csv")
                    header = ["id_x", "id_y", "hospital", "diagnosis", "euclidean_distance", "euclidean_similarity"]
                    euclidean_file.write_to_csv_file(header)
            else:
                self.created_euclidean_file = True
        
        # connect rows
        
        x = np.array(df.iloc[connections[0], 0:34]).astype(float).reshape(-1,1)
        y = np.array(df.iloc[connections[1], 0:34]).astype(float).reshape(-1,1)
    

        # Calculating euclidean distance
        eu_distance = euclidean_distance.Euclidean_Distance()
        euclidean_value = eu_distance.measure_distance(x, y)
        euclidean_similarity_value = eu_distance.calculate_similarity(euclidean_value)

        # Caluclating CRE
        cre_x = cre.CRE(x)
        cre_x_value = cre_x.cre_gaussian_distribution()
        cre_y = cre.CRE(y)
        cre_y_value = cre_y.cre_gaussian_distribution()

        #calculate expecation value X|Y
        mean_y = np.mean(y)
        sigma_y = np.std(y)

        new_data= np.concatenate((x, y), axis = 1)

        ccre_distance = ccre.CCRE(new_data.T)
        cov_conditional_dist = ccre_distance.cov_conditional_distribution() 
        expect_value_cre_xy = integrate.dblquad(ccre_distance.calculate_expectation_value_xy, -np.inf, np.inf, np.mean(x), np.inf, args=(mean_y, sigma_y, cov_conditional_dist))

        ccre_value = (cre_x_value + (expect_value_cre_xy[0]))/cre_x_value
        if ccre_value < 0:
            
            ccre_value = [0, ccre_value]
        

        # write to csv files
        center_label_x = df.iloc[connections[0]]['center_label']
        center_label_y = df.iloc[connections[1]]['center_label']
        diagnosis_label_x = df.iloc[connections[0]]['diagnosis_label']
        diagnosis_label_y = df.iloc[connections[1]]['diagnosis_label']
        ccre_data = [df.index[connections[0]], df.index[connections[1]], f'{center_label_x}_{center_label_y}', f'{diagnosis_label_x}_{diagnosis_label_y}', cre_x_value, cre_y_value, expect_value_cre_xy[0], ccre_value]
        euclidean_data = [df.index[connections[0]], df.index[connections[1]], f'{center_label_x}_{center_label_y}', f'{diagnosis_label_x}_{diagnosis_label_y}', euclidean_value, euclidean_similarity_value]
        with lock:
            if self.created_cre_file:
                
                    with open(f"CCRE_distances_adasyn.csv", "a", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerow(ccre_data)
            else:
            
                ccre_file.write_to_csv_file(ccre_data)
        
            if self.created_euclidean_file:
                
                    with open(f"euclidean_distances_adasyn.csv", "a", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerow(euclidean_data)
            else:
                
                    euclidean_file.write_to_csv_file(euclidean_data)
    
        
        
hospitals = ["UMCG", "CUN", "UGOSM"]
diagnosis = ["HC", "AD", "PD"]


if __name__ == "__main__":
    pool = mp.Pool(20)
    m = mp.Manager()
    l = m.Lock()

    main_class = Main(hospital=hospitals, diagnosis=diagnosis)
    connections = main_class.get_neighbours()
    oversampled_df = main_class.load_data()
    func = partial(main_class.main, l, oversampled_df)
    pool.map(func, connections)
