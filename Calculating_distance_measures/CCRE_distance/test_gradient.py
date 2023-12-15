import numpy as np
import random
import cre 
import ccre
import scipy.integrate as integrate
import matplotlib.pyplot as plt

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


# def test_gradientfunction(data, prototype, percentage=0.5):
        
      
#         length_data = len(data)
#         print(prototype)
#         random_order = random.sample(range(length_data), length_data)
        
#         for item in random_order:
           
#             row = np.array(data[item]).reshape(-1,1)
#             prototype = np.array(prototype).reshape(-1,1)
#             original_data =np.concatenate((row, prototype), axis=1)
#             # original_data = np.array(original_data)
#             new_prototype = []
#             for i in original_data:
               
#                 if i[0] < i[1]:
                    
#                     difference = i[1] - i[0]
            
#                 elif i[0] > i[1]:
#                     difference = -(i[0] - i[1])

#                 else:
#                     print('equal')
#                 difference = difference * percentage
            
#                 new_prototype.append(i[1]-difference)
#             prototype = new_prototype   
        
#         return prototype
# data = [list(np.random.normal(0, 1, 35)) for _ in range (35)]
# prototype = [0 for _ in range(35)]
# print(test_gradientfunction(data, prototype))

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
                      
            x = original_data[:2, 0][0]
            y = original_data[:2, 0][1]
            
            original_ccre_value = calculate_ccre(original_data, row, prototype)
            
            prototype = get_new_prototype(original_data, row, prototype, original_ccre_value, difference_in_ccre)
            
            
            plt.scatter(x, y, color='green')
        x = prototype[0]
        y = prototype[1]
        plt.scatter(x,y, color ='black')
        plt.title('test gradient function')
        plt.xlabel('first value in list')
        plt.ylabel('second value in list')
        plt.show()
        return prototype
data = [list(np.random.normal(0, 1, 35)) for _ in range (10)]
prototype = [np.random.normal(0, 1, 35)]
for _ in range(10):
    prototype = test_gradientfunction_with_ccre(data, prototype)
    print(prototype)