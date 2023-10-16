
import sys
sys.path.append('/home/fcourse1/Desktop/afstudeerstage/code/CCRE_LVQ_project/Calculating_distance_measures/data')


import load_data
sys.path.append('/home/fcourse1/Desktop/afstudeerstage/code/CCRE_LVQ_project/Calculating_distance_measures/euclidean_distance')
import euclidean_distance
import numpy as np
sys.path.append('/home/fcourse1/Desktop/afstudeerstage/code/CCRE_LVQ_project/Calculating_distance_measures/CCRE_distance')
import cre 
import ccre
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
        
        #TO TEST CORRELATION OF 2 DATASETS
        x = np.array([[-0.06800444],
        [ 0.16943253],
        [ 2.42000786],
        [-0.43541997],
        [-0.8019287 ],
        [-0.97924357],
        [ 1.85310014],
        [-0.11858712],
        [-0.29550731],
        [-1.46707809],
        [-0.03476563],
        [-1.42155007],
        [ 0.2448333 ],
        [ 0.8538168 ],
        [ 0.9727146 ],
        [-0.66576435],
        [-0.48732568],
        [ 0.15018597],
        [ 0.17948517],
        [-0.039039  ]])


        y = np.array([[ 0.22147024],
        [ 0.83070095],
        [ 0.89207153],
        [-0.70552404],
        [ 0.72734723],
        [-1.35638024],
        [-0.85044476],
        [ 1.1530766 ],
        [ 2.22923693],
        [-1.44447676],
        [-0.51559235],
        [ 0.83248201],
        [ 0.60668853],
        [ 0.91990303],
        [ 1.64502999],
        [-0.51957907],
        [ 0.78325817],
        [ 1.7536485 ],
        [ 0.97921261],
        [ 0.42252685]])
        original_data = np.concatenate((x,y), axis=1)
        # print(data[:, 0].shape)

      
        for i in np.linspace(0,1, 10):
            i = round(i, 2)
            print(i)
            M = np.array([[1,i], [i,1]])
            data = np.dot(original_data, M)
            x = data[:, 0].reshape(-1,1)
            y = data[:,1].reshape(-1,1)
            
            mean_x = np.mean(x)
            sigma2 = np.var(x)
            sigma = np.sqrt(sigma2)
            
            data = np.concatenate((y,x), axis=1)
         
            cre_distance = cre.CRE(x)
            cre_distance.cre_gaussian_distribution()
            ccre_distance = ccre.CCRE(data.T)
          
            
            cov_conditional_dist = ccre_distance.cov_conditional_distribution() 
            
            print(integrate.dblquad(ccre_distance.calculate_expactation_value, -np.inf, np.inf, 0, np.inf, args=(mean_x, sigma, sigma2, cov_conditional_dist, cre_distance)))
            break
        
        #TO TEST X|Y AND Y|X
        # for i in range(len(df)):
        #     cre_distance = cre.CRE(df.iloc[i, 2:-1])
        #     # cre_distance.calculate_cre()
        #     cre_distance.cre_gaussian_distribution()

        #     # #calculating CCRE 
        #     x = df.iloc[i, 2:-1]
        #     y = df.iloc[i, 2:-1]
        #     x = np.array(x).reshape(-1,1)
        #     y =np.array(y).reshape(-1,1)
        #     x= x.astype(np.float32)
        #     y = y.astype(np.float32)

        #     mean_x = np.mean(x)
        #     sigma2 = np.var(x)
        #     sigma = np.sqrt(sigma2)

        #     data = np.concatenate((y,x), axis=1)
        #     ccre_distance = ccre.CCRE(data.T)
         
        #     cov_conditional_dist = ccre_distance.cov_conditional_distribution() 
        
        #     print(integrate.dblquad(ccre_distance.calculate_expactation_value, -np.inf, np.inf, 0, np.inf, args=(mean_x, sigma, sigma2, cov_conditional_dist, cre_distance)))
            
        #     #test_x given y
        #     mean_y = np.mean(y)
        #     sigma2_y = np.var(y)
        #     sigma_y = np.sqrt(sigma2_y)
        #     data = np.concatenate((x,y), axis=1)
        #     ccre_distance = ccre.CCRE(data.T)
        #     cov_conditional_dist = ccre_distance.cov_conditional_distribution() 
        #     print(integrate.dblquad(ccre_distance.calculate_expectation_value_xy, -np.inf, np.inf, 0, np.inf, args=(mean_y, sigma_y, sigma2_y, cov_conditional_dist, cre_distance)))
        #     print(i)
        return "lets stop here"
        range_of_y = np.sqrt(np.var(y)) * 4 #range for new entries to calculate the cre of Y|X X will remain constant
        # print(range_of_y)
        new_dx = np.linspace(0,0,1000000) #will remain a fixed value
        new_dy = np.linspace(-range_of_y, range_of_y, 1000000) #between 4 * standard deviation of y
        # new_entry = np.array([0, 0])
        bivariate_PDF = []
        for index, _ in enumerate(new_dx):
            new_entry = np.array([new_dy[index], 0])
            # print(ccre_distance.mean[0], ccre_distance.mean[1])
            result = ccre_distance.calculate_joint_dist(new_entry)
            bivariate_PDF.append(result)
        cre_YX = cre.CRE(bivariate_PDF)
        cre_YX.cre_gaussian_distribution()
        
        # # test calculate marginal
        # marginal = ccre_distance.calculate_margin_pdf(result, np.mean(x), np.sqrt(np.var(x)), np.var(x))
        # print(marginal)
  
    
   
        
     
        
hospitals = ["UMCG", "CUN", "UGOSM"]
diagnosis = ["HC", "AD", "PD"]

# for hospital in hospitals:
#     for diagnosi in diagnosis:
#         if (hospital == "CUN") & (diagnosi == "AD"):
#             continue
#         print(f"{hospital} {diagnosi}: {Main(hospital, diagnosi).main()}")
Main("UMCG", "AD").main()
