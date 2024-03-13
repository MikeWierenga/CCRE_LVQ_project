import numpy as np
import metric
import random
class LVQ:
    def __init__(self) -> None:
        pass

    def train(self, epoch, dataset, prototype, metric, difference_in_ccre=0.05):
        new_prototype = np.array(prototype).reshape(-1,1).astype(float)
        n_data = len(dataset)
        
        prototypes = []
        for _ in range(epoch):
            random_order = random.sample(range(n_data), n_data)
            for item in random_order:
                row = np.array(dataset.iloc[item]).reshape(-1,1)
                
                original_data =np.concatenate((row[2:-1].astype(float), new_prototype), axis=1).astype(float)    
                
            
                original_ccre_value = metric.calculate_ccre(original_data, row[2:-1], prototype)
                
                new_prototype = np.array(self.get_new_prototype(metric, original_data, row[2:-1], prototype, original_ccre_value, difference_in_ccre)).reshape(-1,1)
            prototypes.append(new_prototype)     
        return prototypes
        
    
    def get_new_prototype(self, metric, original_data, row, prototype, original_ccre_value, difference_in_ccre):
      for percentage in np.linspace(0, 1, 20):
        new_prototype = []
        for i in original_data:
        
            if i[0] < i[1]:
                
                difference = i[1] - i[0]
        
            elif i[0] > i[1]:
                difference = -(i[0] - i[1])

            else:
                print('equal')
            difference = difference * percentage
        
            new_prototype.append(i[1]-difference)
        prototype = np.array(new_prototype).reshape(-1,1)
        

        new_data = np.concatenate((row, prototype), axis=1)
        new_ccre_value = metric.calculate_ccre(new_data.astype(float), row, prototype)
     
        if (new_ccre_value - original_ccre_value) >= difference_in_ccre:
                
                return prototype

    def create_prototype(self, dimensions):
        prototype = np.random.normal(0,1, dimensions)
        return prototype
    
    def fit():
        pass

    def predict(self, test_df, prototypes, metric):
        
        values = test_df.iloc[:, 2:-1].values
        
        predictions = []
        for index, value in enumerate(values):
        
            actual  = test_df.iloc[index]['diagnosis_label']
        
            value = np.array(value).reshape(-1,1)
            HC_value = np.array(prototypes["HC"]).reshape(-1,1)
            combined = np.concatenate((value, HC_value), axis=1)
            HC_CCRE = metric.calculate_ccre(combined.astype(float), value, HC_value)

            AD_value = np.array(prototypes["AD"]).reshape(-1,1)
            combined = np.concatenate((value, AD_value), axis=1)
            AD_CCRE = metric.calculate_ccre(combined.astype(float), value, AD_value)

            PD_value = np.array(prototypes["PD"]).reshape(-1,1)
            combined = np.concatenate((value, PD_value), axis=1)
            PD_CCRE = metric.calculate_ccre(combined.astype(float), value, PD_value)
            test = {"HC":HC_CCRE, "AD":AD_CCRE, "PD":PD_CCRE}
            
            predictions.append([actual, test, max(test, key=test.get), str(actual ==  max(test, key=test.get))])
        return predictions