import numpy as np
import random
import cre 
import ccre
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import sys
from collections import Counter
sys.path.append('data')


import load_data

def calculate_cre(data):
        cre_x = cre.CRE(data)
        cre_value_x = cre_x.cre_gaussian_distribution()
        return cre_value_x

def calculate_expectation_value_xy(data, mean_y, sigma):
        ccre_xy = ccre.CCRE(data)
        cov_conditional_dist = ccre_xy.cov_conditional_distribution()
        expected_value_cre_xy = integrate.dblquad(ccre_xy.calculate_expectation_value_xy, -np.inf, np.inf, 0, np.inf, args=(mean_y, sigma, cov_conditional_dist))
        return expected_value_cre_xy


def calculate_ccre(data, x, y):
    old_cre_value = calculate_cre(x)
    old_cre_y_value = calculate_cre(y)


    if old_cre_value == old_cre_y_value:
        old_expected_value_cre_xy = [0]
    else:
        old_expected_value_cre_xy = calculate_expectation_value_xy(data.T, np.mean(y), np.std(y))

    return (old_cre_value + old_expected_value_cre_xy[0]) / old_cre_value


def load_the_data():
        # loading the data
        data = load_data.Load_Data()
        feature_vectors = data.get_dataframe('feature_vectors')
        feature_vectors = feature_vectors.set_axis(list(range(35)), axis=1)
        center = data.get_dataframe('center_label')
        center.rename(columns={0: "center_label"}, inplace=True)

        diagnosis = data.get_dataframe('diagnosis_label')
        diagnosis.rename(columns={0: "diagnosis_label"}, inplace=True)

        df = data.combine_dataframes(center, diagnosis, feature_vectors)
        
        patients_to_remove = [23, 25, 27,30,32,34,35,36,40,52, 56, 104,110,111,117,120,134,140,148,159,161,162, 171,179,180,181,182,191,192,196,205,211,215,216, 230,233,236,237,238,239,242,243,245, 247,252,253,254,256,258,262,272,275,277,279,284,285,286,287,289,290,291,292,294,295,296,297,298,299,300,301,303]
        df = df.drop(patients_to_remove)
       
    
        return df

def split_data(df):
    #splitting the data
    df = df.reset_index(drop=True)
    df['centre_diagnosis'] = df['center_label'] + '_' + df['diagnosis_label']
    

    test_ids = []
    for key, value in Counter(df['centre_diagnosis']).items():
        for id in df.index[df['centre_diagnosis'] == key].tolist()[:round(value/100*20)]:
            test_ids.append(id)
    
    testdf =df.loc[df.index[test_ids]]
    traindf = df.loc[~df.index.isin(test_ids)]
    return testdf, traindf

def get_new_prototype(original_data, row, prototype, original_ccre_value, difference_in_ccre):
      for percentage in np.linspace(0, 1, 10):
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
        new_ccre_value = calculate_ccre(new_data, row, prototype)
     
        if (new_ccre_value - original_ccre_value) >= difference_in_ccre:
                print(new_ccre_value, original_ccre_value)
                return prototype
        else:
            print('i got here', new_ccre_value - original_ccre_value)

def test_gradientfunction_with_ccre(data, prototype, difference_in_ccre=0.1):
        length_data = len(data)
        random_order = random.sample(range(length_data), length_data)
        prototype = np.array(prototype).reshape(-1,1)
        
        for item in random_order:
            row = np.array(data[item]).reshape(-1,1)
            original_data =np.concatenate((row, prototype), axis=1)    
            original_ccre_value = calculate_ccre(original_data, row, prototype)
            prototype = get_new_prototype(original_data, row, prototype, original_ccre_value, difference_in_ccre)

        return prototype

def predict(test_df, prototypes):
    values = test_df.iloc[:, 2:-1].values
    predictions = []
    for index, value in enumerate(values):
        
        actual  = test_df.iloc[index]['diagnosis_label']
       
        value = np.array(value).reshape(-1,1)
        HC_value = np.array(prototypes["HC"]).reshape(-1,1)
        combined = np.concatenate((value, HC_value), axis=1)
        HC_CCRE = calculate_ccre(combined, value, HC_value)

        AD_value = np.array(prototypes["AD"]).reshape(-1,1)
        combined = np.concatenate((value, AD_value), axis=1)
        AD_CCRE = calculate_ccre(combined, value, AD_value)

        PD_value = np.array(prototypes["PD"]).reshape(-1,1)
        combined = np.concatenate((value, PD_value), axis=1)
        PD_CCRE = calculate_ccre(combined, value, PD_value)
        test = {"HC":HC_CCRE, "AD":AD_CCRE, "PD":PD_CCRE}
        
        predictions.append([actual, test, max(test, key=test.get), str(actual ==  max(test, key=test.get))])
    return predictions


def train_prototype(prototype, train_data):
    
    prototype = test_gradientfunction_with_ccre(train_data, prototype)
    return prototype

# data = load_the_data("HC")
# test_df, train_df = split_data(data)
# prototype = [np.random.normal(0, 1, 35)]
# train_df_values = train_df.iloc[:, 2:-1].values
# prototype = train_prototype(prototype, train_df_values)
# print(prototype)

def scatter_plot(train_data, prototype_data):
    diagnosis_labels = ["HC", "AD", "PD"]
    diagnosis_color = {"HC": "gray", "AD": "Blue", "PD": "violet"}
    prototype_color = {"HC": "black", "AD": "navy", "PD": "purple"}
    for diagnosis in  diagnosis_labels:
            x = train_data[diagnosis].values[:, 2:3]
            y = train_data[diagnosis].values[:, 3:4]
            p_x = prototype_data[diagnosis][0][0]
            p_y = prototype_data[diagnosis][1][0]

            plt.scatter(x, y, color=diagnosis_color[diagnosis], label=diagnosis, )
            plt.scatter(p_x, p_y, color=prototype_color[diagnosis], label=f"prototype_{diagnosis}", s=100) 
    plt.legend()
    plt.show()

def bar_chart(data):
    courses = list(data.keys())
    values = list(data.values())
    print(courses)
    fig = plt.figure(figsize = (10, 5))
    
    # creating the bar plot
    plt.bar(courses, values, color ='maroon', 
            width = 0.4)
    
    plt.xlabel("Predicted")
    plt.ylabel("Amount")
    plt.title("Predictions using LVQ with CCRE")
    plt.show()

def train_model(epoch=10):
    prototype_HC = [np.random.normal(0, 1, 35)]
    prototype_AD = [np.random.normal(0, 1, 35)]
    prototype_PD = [np.random.normal(0, 1, 35)]
    data = load_the_data()
    test_df, train_df = split_data(data)
    
    for i in range(epoch):
        # HC
        HC_df = train_df[train_df['diagnosis_label'] == "HC"]
        HC_df_values = HC_df.iloc[:, 2:-1].values
        prototype_HC = train_prototype(prototype_HC, HC_df_values)
        print("HC DONE")
        # AD
        AD_df = train_df[train_df['diagnosis_label'] == "AD"]
        AD_df_values = AD_df.iloc[:, 2:-1].values
        prototype_AD = train_prototype(prototype_AD, AD_df_values)
        print("AD DONE")
        # PD
        PD_df = train_df[train_df['diagnosis_label'] == "PD"]
        PD_df_values = PD_df.iloc[:, 2:-1].values
        prototype_PD = train_prototype(prototype_PD, PD_df_values)
        print("PD DONE")
        final_prototypes = {"HC": prototype_HC, "AD": prototype_AD, "PD": prototype_PD}
        train_values = {"HC": HC_df, "AD": AD_df, "PD": PD_df}
        all_predictions = predict(test_df, final_prototypes)
        results = np.array(all_predictions)
        bar_chart(Counter(results[:, 3]))
        scatter_plot(prototype_data=final_prototypes, train_data=train_values)
        

train_model()







