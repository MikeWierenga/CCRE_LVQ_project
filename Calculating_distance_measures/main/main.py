import sys
sys.path.append('data')
import load_data
sys.path.append('euclidean_distance')
import euclidean_distance
import numpy as np
sys.path.append('CCRE_distance')
import cre 
import ccre
import pandas as pd
from sklearn import preprocessing
import scipy.integrate as integrate
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
        
        #normalizing the data(everything is between 0 and 1)
        df_labels = df.iloc[:, :2]
        x = df.iloc[:, 2:].values
        scaler = preprocessing.MinMaxScaler(feature_range=(0,1)).fit_transform(x)

        df_scaled = pd.DataFrame(scaler)
        
        dataframes = [df_labels, df_scaled]
        df = pd.concat(dataframes, axis=1)

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
        cre_distance = cre.CRE(df)
        cre_distance.calculate_cre()
        cre_distance.cre_gaussian_distribution()

        # #calculating CCRE 
        print("VANAF HIER IS CCRE")
        ccre_distance = ccre.CCRE()
        x = df.iloc[0,2:]
        y = df.iloc[1, 2:]
        x = np.array(x).reshape(-1,1)
        y =np.array(y).reshape(-1,1)
        data = np.concatenate((x,y), axis=1)
   
        
        joint = ccre_distance.calculate_joint_dist(data)
        print(joint)
    
        # mu = np.mean(data)
        # sigma2 = np.var(data)
        # sigma = np.sqrt(sigma2)
        
        # result = integrate.quad(ccre_distance.calculate_expactation_value, 0, np.inf, args=(mu, sigma, sigma2))
        # print(result)
        
hospitals = ["UMCG", "CUN", "UGOSM"]
diagnosis = ["HC", "AD", "PD"]

# for hospital in hospitals:
#     for diagnosi in diagnosis:
#         if (hospital == "CUN") & (diagnosi == "AD"):
#             continue
#         print(f"{hospital} {diagnosi}: {Main(hospital, diagnosi).main()}")
Main("UMCG", "AD").main()
