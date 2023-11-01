
import sys
import os
import csv
sys.path.append('/home/fcourse1/Desktop/afstudeerstage/code/CCRE_LVQ_project/Calculating_distance_measures/data')


import load_data
sys.path.append('/home/fcourse1/Desktop/afstudeerstage/code/CCRE_LVQ_project/Calculating_distance_measures/euclidean_distance')
import euclidean_distance
import numpy as np
sys.path.append('/home/fcourse1/Desktop/afstudeerstage/code/CCRE_LVQ_project/Calculating_distance_measures/CCRE_distance')
import cre 
import ccre
sys.path.append('/home/fcourse1/Desktop/afstudeerstage/code/CCRE_LVQ_project/Calculating_distance_measures/neighbours')
import neighbours
sys.path.append('/home/fcourse1/Desktop/afstudeerstage/code/CCRE_LVQ_project/Calculating_distance_measures/write_to_csv')
import csv_file
import pandas as pd
from sklearn import preprocessing
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from scipy.stats import norm

class Main():

    def __init__(self, hospital, diagnosis):
        self.hospital = hospital
        self.diagnosis = diagnosis
        self.created_cre_file = False
        self.created_euclidean_file = False
        
    def main(self):
        # loading the data
        data = load_data.Load_Data()
        feature_vectors = data.get_dataframe('feature_vectors')
        feature_vectors = feature_vectors.set_axis(list(range(1, 36)), axis=1)
        center = data.get_dataframe('center_label')
        center.rename(columns={0: "center_label"}, inplace=True)

        diagnosis = data.get_dataframe('diagnosis_label')
        diagnosis.rename(columns={0: "diagnosis_label"}, inplace=True)

        df = data.combine_dataframes(center, diagnosis, feature_vectors)
        df = df[(df['center_label'].isin(self.hospital)) & (df['diagnosis_label'].isin(self.diagnosis))]
        
        df = data.limit_dataframe(df, 10)
        
        # create csv file
        if not os.path.isfile(f"CCRE_distances_{self.hospital}_{self.diagnosis}.csv"):
            
            ccre_file = csv_file.CSV(f"CCRE_distances_{self.hospital}_{self.diagnosis}.csv")
            header = ["id_x", "id_y", "hospital", "diagnosis", "cre(x)", "cre(y)", "E[cre(X|Y)]", "ccre(X|Y)"]
            ccre_file.write_to_csv_file(header)
        else:
        
            self.created_cre_file = True
        if not os.path.isfile(f"euclidean_distances_{self.hospital}_{self.diagnosis}.csv"):
            
            euclidean_file = csv_file.CSV(f"euclidean_distances_{self.hospital}_{self.diagnosis}.csv")
            header = ["id_x", "id_y", "hospital", "diagnosis", "euclidean_distance", "euclidean_similarity"]
            euclidean_file.write_to_csv_file(header)
        else:
            self.created_euclidean_file = True
         
        # connect rows
        connection = neighbours.neighbours(df)
        for i in connection.connect_neighbours():
            print(i)
            x = np.array(df.iloc[i[0], 2:-1]).astype(float).reshape(-1,1)
            y = np.array(df.iloc[i[1], 2:-1]).astype(float).reshape(-1,1)
            
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
            expect_value_cre_xy = integrate.dblquad(ccre_distance.calculate_expectation_value_xy, -np.inf, np.inf, 0, np.inf, args=(mean_y, sigma_y, cov_conditional_dist))
   
            ccre_value = (cre_x_value + (expect_value_cre_xy[0]))/cre_x_value
            if ccre_value < 0:
              
                ccre_value = abs(ccre_value)
            
            
            # write to csv files
            center_label_x = df.iloc[i[0]]['center_label']
            center_label_y = df.iloc[i[1]]['center_label']
            diagnosis_label_x = df.iloc[i[0]]['diagnosis_label']
            diagnosis_label_y = df.iloc[i[1]]['diagnosis_label']
            ccre_data = [df.index[i[0]], df.index[i[1]], f'{center_label_x}_{center_label_y}', f'{diagnosis_label_x}_{diagnosis_label_y}', cre_x_value, cre_y_value, expect_value_cre_xy[0], ccre_value]
            euclidean_data = [df.index[i[0]], df.index[i[1]], f'{center_label_x}_{center_label_y}', f'{diagnosis_label_x}_{diagnosis_label_y}', euclidean_value, euclidean_similarity_value]
            if self.created_cre_file:
                with open("CCRE_distances.csv", "a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(ccre_data)
            else:
                
                ccre_file.write_to_csv_file(ccre_data)
            
            if self.created_euclidean_file:
                with open("euclidean_distances.csv", "a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(euclidean_data)
            else:
                euclidean_file.write_to_csv_file(euclidean_data)
        
        
        
hospitals = ["UMCG", "CUN", "UGOSM"]
diagnosis = ["HC", "AD", "PD"]

# for hospital in hospitals:
#     for diagnosi in diagnosis:
#         if (hospital == "CUN") & (diagnosi == "AD"):
#             continue
#         Main(hospital, diagnosi).main()

Main(hospitals, diagnosis).main()

