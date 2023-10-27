import yaml
import pandas as pd
import os
class Load_Data:
    def __init__(self):
        self.config = self.get_config()
    
    def get_config(self):
        with open("Calculating_distance_measures/main/config.yaml", "r") as stream:
            config = yaml.safe_load(stream)
        return config
    
    def combine_dataframes(self, *dataframes):
        return pd.concat(dataframes, axis=1)

    def get_dataframe(self, filename):
        df = pd.read_csv(self.config[filename], header = None)
        return df

    def limit_dataframe(self, df, amount):
        list_of_df = []
        for center in df.center_label.unique():
            for diagnosis in df.diagnosis_label.unique():
                list_of_df.append(df[(df.center_label == center) & (df.diagnosis_label == diagnosis)][:amount] )
            
        return pd.concat(list_of_df)
# test = Load_Data()
# feature_vectors = test.get_dataframe('feature_vectors')
# feature_vectors = feature_vectors.set_axis(list(range(1, 36)), axis=1)
# center = test.get_dataframe('center_label')
# center.rename(columns={0: "center_label"}, inplace=True)

# diagnosis = test.get_dataframe('diagnosis_label')
# diagnosis.rename(columns={0: "diagnosis_label"}, inplace=True)

# df = test.combine_dataframes(center, diagnosis, feature_vectors)
# print(test.limit_dataframe(df, 10))