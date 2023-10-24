
import sys
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
import pandas as pd
from sklearn import preprocessing
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from scipy.stats import norm

class Main():

    def __init__(self, hospital, diagnosis):
        self.hospital = hospital
        self.diagnosis = diagnosis
    

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
        df = df[(df['center_label'] == self.hospital) & (df['diagnosis_label'] == self.diagnosis)]
        #normalizing the data(everything is between 0 and 1)
        # df_labels = df.iloc[:, :2]
        # x = df.iloc[:, 2:].values
        # scaler = preprocessing.MinMaxScaler(feature_range=(0,1)).fit_transform(x)

        # df_scaled = pd.DataFrame(scaler)
        
        # dataframes = [df_labels, df_scaled]
        # df = pd.concat(dataframes, axis=1)

        # # calculating the Euclidean distance
        # eu_distance = euclidean_distance.Euclidean_Distance(self.hospital, self.diagnosis)

        # # get dataframe that only consists of with the labels mentioned above
        # specific_df = df[(df['center_label'] == eu_distance.hospital) & (df['diagnosis_label'] == eu_distance.diagnosis)]

        # # connect all rows with eachother and make sure there are no duplicates
        # connections = eu_distance.connect_rows(specific_df)
        # distances = []
        # for i in connections:
        #     position1 = np.array(specific_df.iloc[i[0]][2:])
        #     position2 = np.array(specific_df.iloc[i[1]][2:])
        #     distances.append(eu_distance.measure_distance(position1, position2))
        # print(distances)
        # # calculate average distance
        # average_distance = eu_distance.calculate_average(distances)
        # return average_distance

        # calculate the CRE
        connection = neighbours.neighbours(df)
        ccre_distances_values = []
        for i in connection.connect_neighbours():
            print(i)
            x = np.array(df.iloc[i[0], 2:-1]).astype(float).reshape(-1,1)
            y = np.array(df.iloc[i[1], 2:-1]).astype(float).reshape(-1,1)
            cre_x = cre.CRE(x)
            cre_x_value = cre_x.cre_gaussian_distribution()
            cre_y = cre.CRE(y)
            cre_y_value = cre_y.cre_gaussian_distribution()
            print(f'cre(x) = {cre_x_value} cre(y) = {cre_y_value}')
            
            
            #calculate expectation value Y|X
            # mean_x = np.mean(x)
            # sigma = np.std(x)
            # sigma2 = np.var(x)

            # new_data= np.concatenate((x, y), axis = 1)

            # ccre_distance = ccre.CCRE(new_data.T)
            # cov_conditional_dist = ccre_distance.cov_conditional_distribution() 
            # expect_value_cre_yx = integrate.dblquad(ccre_distance.calculate_expectation_value, -np.inf, np.inf, 0, np.inf, args=(mean_x, sigma, sigma2, cov_conditional_dist, cre_x))
            # print(f'E[e[Y|X]] = {expect_value_cre_yx}')
        
            # print(f"ccre(X - Y|X) = {(cre_x_value + (expect_value_cre_yx[0]))/cre_x_value}\n")
                
            #calculate expecation value X|Y
            mean_y = np.mean(y)
            sigma_y = np.std(y)
            sigma2_y = np.var(y)

            new_data= np.concatenate((x, y), axis = 1)

            ccre_distance = ccre.CCRE(new_data.T)
            cov_conditional_dist = ccre_distance.cov_conditional_distribution() 
            expect_value_cre_xy = integrate.dblquad(ccre_distance.calculate_expectation_value_xy, -np.inf, np.inf, 0, np.inf, args=(mean_y, sigma_y, sigma2_y, cov_conditional_dist, cre_x))
            print(f'E[e[X|Y]] = {expect_value_cre_xy}')
            print(f"ccre(X - X|Y) = {(cre_x_value + (expect_value_cre_xy[0]))/cre_x_value}\n")
            ccre_value = (cre_x_value + (expect_value_cre_xy[0]))/cre_x_value
            if ccre_value < 0:
                print(ccre_value)
                ccre_value = 0
            ccre_distances_values.append(ccre_value)
        return sum(ccre_distances_values)/len(ccre_distances_values)
        x = np.array(df.iloc[2, 2:-1]).astype(float).reshape(-1,1)
        y = np.array(df.iloc[4, 2:-1]).astype(float).reshape(-1,1)
        cre_x = cre.CRE(x)
        cre_x_value = cre_x.cre_gaussian_distribution()
        cre_y = cre.CRE(y)
        cre_y_value = cre_y.cre_gaussian_distribution()
        print(f'cre(x) = {cre_x_value} cre(y) = {cre_y_value}')
        
        
        #calculate expectation value Y|X
        mean_x = np.mean(x)
        sigma = np.std(x)
        sigma2 = np.var(x)

        new_data= np.concatenate((x, y), axis = 1)

        ccre_distance = ccre.CCRE(new_data.T)
        cov_conditional_dist = ccre_distance.cov_conditional_distribution() 
        expect_value_cre_yx = integrate.dblquad(ccre_distance.calculate_expectation_value, -np.inf, np.inf, 0, np.inf, args=(mean_x, sigma, sigma2, cov_conditional_dist, cre_x))
        print(f'E[e[Y|X]] = {expect_value_cre_yx}')
       
        print(f"ccre(X - Y|X) = {(cre_x_value + (expect_value_cre_yx[0]))/cre_x_value}\n")
            
        #calculate expecation value X|Y
        mean_y = np.mean(y)
        sigma_y = np.std(y)
        sigma2_y = np.var(y)

        new_data= np.concatenate((x, y), axis = 1)

        ccre_distance = ccre.CCRE(new_data.T)
        cov_conditional_dist = ccre_distance.cov_conditional_distribution() 
        expect_value_cre_xy = integrate.dblquad(ccre_distance.calculate_expectation_value_xy, -np.inf, np.inf, 0, np.inf, args=(mean_y, sigma_y, sigma2_y, cov_conditional_dist, cre_x))
        print(f'E[e[Y|X]] = {expect_value_cre_xy}')
        print(f"ccre(X - Y|X) = {(cre_x_value + (expect_value_cre_xy[0]))/cre_x_value}\n")

        plt.scatter(x, y)
        plt.show()
    
       
  
    
   
        
     
        
hospitals = ["UMCG", "CUN", "UGOSM"]
diagnosis = ["HC", "AD", "PD"]

# for hospital in hospitals:
#     for diagnosi in diagnosis:
#         if (hospital == "CUN") & (diagnosi == "AD"):
#             continue
#         print(f"{hospital} {diagnosi}: {Main(hospital, diagnosi).main()}")
Main("UMCG", "AD").main()
