import sys
sys.path.append('data')
import load_data
sys.path.append('euclidean_distance')
import euclidean_distance
import numpy as np
sys.path.append('CCRE_distance')
import cre 
import pandas as pd

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
        normalized_df =  (df.iloc[:, 2:]-df.iloc[:, 2:].mean())/df.iloc[:, 2:].std()
        # normalized_df = (df.iloc[:, 2:]-df.iloc[:, 2:].min())/(df.iloc[:, 2:].max()-df.iloc[:, 2:].min())
        dataframes = [df_labels, normalized_df]
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

        #calculating CCRE 
        
hospitals = ["UMCG", "CUN", "UGOSM"]
diagnosis = ["HC", "AD", "PD"]

# for hospital in hospitals:
#     for diagnosi in diagnosis:
#         if (hospital == "CUN") & (diagnosi == "AD"):
#             continue
#         print(f"{hospital} {diagnosi}: {Main(hospital, diagnosi).main()}")
Main("UMCG", "AD").main()
