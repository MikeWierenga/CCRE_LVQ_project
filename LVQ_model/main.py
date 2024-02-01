import sys
import pandas as pd
sys.path.append('data')
import model
import load_data
from collections import Counter
import metric
import numpy as np
import visualizations

class Main:
  
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
        patients_to_remove = [23, 25, 27,30,32,34,35,36,40,52, 56, 104,110,111,117,120,134,140,148,159,161,162, 171,179,180,181,182,191,192,196,205,211,215,216, 230,233,236,237,238,239,242,243,245, 247,252,253,254,256,258,262,272,275,277,279,284,285,286,287,289,290,291,292,294,295,296,297,298,299,300,301,303]
        df = df.drop(patients_to_remove)
        df = df.reset_index(drop=True)
        return df
    
    def split_data(self, df):
        df['centre_diagnosis'] = df['center_label'] + '_' + df['diagnosis_label']
    

        test_ids = []
        for key, value in Counter(df['centre_diagnosis']).items():
            for id in df.index[df['centre_diagnosis'] == key].tolist()[:round(value/100*20)]:
                test_ids.append(id)
        
        testdf =df.loc[df.index[test_ids]]
        traindf = df.loc[~df.index.isin(test_ids)]
        return testdf, traindf
    
    def main(self):
        #load data
        data = self.load_data()
        #split data
        test_data, training_data = self.split_data(data)
        
        # create metric class
        ccre_metric = metric.CCRE()
        
        # create LVQ class
        LVQ = model.LVQ()

        #create prototypes
        HC_prototype = LVQ.create_prototype(dimensions=35)
        AD_prototype = LVQ.create_prototype(dimensions=35)
        PD_prototype = LVQ.create_prototype(dimensions=35)
        
        #train prototypes
        epoch = 10
        trained_HC_prototype = LVQ.train(epoch=epoch, dataset=training_data[training_data.diagnosis_label == "HC"], prototype=HC_prototype, metric=ccre_metric)
        trained_AD_prototype = LVQ.train(epoch=epoch, dataset=training_data[training_data.diagnosis_label == "AD"], prototype=AD_prototype, metric=ccre_metric)
        trained_PD_prototype = LVQ.train(epoch=epoch, dataset=training_data[training_data.diagnosis_label == "PD"], prototype=PD_prototype, metric=ccre_metric)
        final_prototypes = {"HC": trained_HC_prototype, "AD": trained_AD_prototype, "PD": trained_PD_prototype}
        
        #create visualization class
        visualization = visualizations.Visualizations()

        
        #predict
        HC_df = training_data[training_data['diagnosis_label'] == "HC"]
        AD_df = training_data[training_data['diagnosis_label'] == "AD"]
        PD_df = training_data[training_data['diagnosis_label'] == "PD"]
        train_values = {"HC": HC_df, "AD": AD_df, "PD": PD_df}
        for i in range(epoch):
            final_prototypes = {"HC": trained_HC_prototype[i], "AD": trained_AD_prototype[i], "PD": trained_PD_prototype[i]}
            predictions = LVQ.predict(test_data, final_prototypes, ccre_metric)
            

            results = np.array(predictions)
            visualization.scatter_plot(train_data=train_values, prototype_data=final_prototypes, epoch=i)
            visualization.bar_chart(Counter(results[:, 3]), i)
            visualization.confussion_matrix(results_matrix=results, epoch=i)


test = Main()
test.main()