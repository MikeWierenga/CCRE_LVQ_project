import yaml
import pandas as pd
class Load_Data:
    def __init__(self):
       self.config = self.get_config()

    def get_config(self):
        with open("main/config.yaml", "r") as stream:
            config = yaml.safe_load(stream)
        return config
    
    def combine_dataframes(self, *dataframes):
        # print(dataframes)
        return pd.concat(dataframes, axis=1)

    def get_dataframe(self, filename):
        df = pd.read_csv(self.config[filename], header = None)
        return df

test = Load_Data()
feature_vectors = test.get_dataframe('feature_vectors')
feature_vectors = feature_vectors.set_axis(list(range(1, 36)), axis=1)
center = test.get_dataframe('center_label')
center.rename(columns={0: "center_label"}, inplace=True)

diagnosis = test.get_dataframe('diagnosis_label')
diagnosis.rename(columns={0: "diagnosis_label"}, inplace=True)

test.combine_dataframes(center, diagnosis, feature_vectors)