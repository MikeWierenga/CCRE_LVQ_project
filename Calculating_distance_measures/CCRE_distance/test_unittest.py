import cre
import ccre
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import unittest


class Test_testcre(unittest.TestCase):
    def setUp(self):
        self.data = np.random.normal(0, 1, 10)
        self.x = np.random.randint(1,10)
        self.mu = np.mean(self.data)
        self.sigma = np.sqrt(np.var(self.data))
        self.cre_class = cre.CRE(self.data)
    
    def test_cumulativedistribution(self):
        self.assertIsInstance(self.cre_class.cumulative_distribution(self.x, self.mu, self.sigma), float)

    def test_cre_gaussian_distribution(self):
        self.assertGreater(self.cre_class.cre_gaussian_distribution(), 0)

    def test_wider_Gaus_distributions(self):    
        before = 0
        for i in np.linspace(0.1, 10, 10):

            dataset = np.random.normal(0,i,1000)
            cre_class = cre.CRE(dataset)
            cre_value = cre_class.cre_gaussian_distribution()
      
            self.assertGreater(cre_value, before)
            
            before = cre_value

    
    # def test_conditional_cre(self):
    #     y = np.random.normal(0,1,10).reshape(-1,1)
    #     x = np.random.normal(0,1, 10).reshape(-1,1)
    #     mean_x = np.mean(x)
    #     sigma = np.std(x)
    #     sigma2 = np.var(x)

    #     original_data = np.concatenate((y, x), axis = 1)
       
    #     # print(f'cre(x) = {cre_x_value} cre(y) = {cre_y_value}')
    #     ccre_class = ccre.CCRE(original_data.T)
    #     cov_conditional_dist = ccre_class.cov_conditional_distribution() 
    #     # print(f'cre(Y|X = 0){integrate.quad(ccre_class.calculate_expactation_value, 0, np.inf, args = (0, mean_x, sigma, sigma2, cov_conditional_dist, cre_class))}')
        
    #     for i in np.linspace(0.01, 1, 100):
    #         print(f'PERCENTAGE = {i}')
    #         print(np.pi/2*i)
    #         m = np.array([[1,0], [np.sin(np.pi/2*i),np.cos(np.pi/2*i)]]).T
    
    #         new_data = np.dot(original_data, m)
    #         new_x = new_data[:, 0].reshape(-1,1)
         
            
    #         mean_x = np.mean(new_x)
    #         sigma = np.std(new_x)
    #         sigma2 = np.var(new_x)

    #         new_y = new_data[:, 1].reshape(-1,1)
    #         new_data= np.concatenate((new_y, new_x), axis = 1)
    #         # print(new_data[0])
    #         cre_class = cre.CRE(new_x)
    #         cre_y = cre.CRE(new_y)
    #         cre_value = cre_class.cre_gaussian_distribution()
    #         cre_y_value = cre_y.cre_gaussian_distribution()
    #         print(f'cre(x) = {cre_value}, cre(y) = {cre_y_value}')

    #         ccre_class = ccre.CCRE(new_data.T)
    #         cov_conditional_dist = ccre_class.cov_conditional_distribution() 
    #         print(cov_conditional_dist)
    #         cov = np.cov(new_data.T)
    #         print(np.linalg.det(cov))
    #         print(cov)
            
    #         print(np.linalg.pinv(cov))
    #         print(f'cre(Y|X = 0){integrate.quad(ccre_class.calculate_expactation_value, 0, np.inf, args = (0, mean_x, sigma, sigma2, cov_conditional_dist, cre_class))} \n')
        
       
    def test_unrelated_to_related_gaussian_distribution(self):
        x = np.random.normal(0,3, 100).reshape(-1,1)
        y = np.random.normal(0,1, 100).reshape(-1,1)
        original_data = np.concatenate((x,y), axis=1)
       
        
        for i in np.linspace(0.01, 1, 10):
            print(f'PERCENTAGE = {i}')
            m = np.array([[1,0], [np.sin(np.pi/2*i),np.cos(np.pi/2*i)]]).T
            
            new_data = np.dot(original_data, m)
          
            new_x = new_data[:, 0].reshape(-1,1)
          
            
            mean_x = np.mean(new_x)
            sigma = np.std(new_x)
            sigma2 = np.var(new_x)
        

            new_y = new_data[:, 1].reshape(-1,1)
           
            
            new_data= np.concatenate((new_y, new_x), axis = 1)
            
            
            cre_class = cre.CRE(new_x)
            cre_y = cre.CRE(new_y)
            cre_y_value = cre_y.cre_gaussian_distribution()
            cre_value = cre_class.cre_gaussian_distribution()
            
            ccre_distance = ccre.CCRE(new_data.T)
            cov_conditional_dist = ccre_distance.cov_conditional_distribution() 
            expect_value_cre_yx = integrate.dblquad(ccre_distance.calculate_expectation_value, -np.inf, np.inf, 0, np.inf, args=(mean_x, sigma, sigma2, cov_conditional_dist, cre_class))
            
            mean_x = np.mean(new_y)
            sigma = np.std(new_y)
            sigma2 = np.var(new_y)
            new_data= np.concatenate((new_x, new_y), axis = 1)
            ccre_distance = ccre.CCRE(new_data.T)
            cov_conditional_dist = ccre_distance.cov_conditional_distribution()
            expected_value_cre_xy = integrate.dblquad(ccre_distance.calculate_expectation_value_xy, -np.inf, np.inf, 0, np.inf, args=(mean_x, sigma, sigma2, cov_conditional_dist, cre_class))
            
            print(f"cre(X) = {cre_value} cre(Y){cre_y_value} E[e(Y|X)]{expect_value_cre_yx[0]} E[e(X|Y)] {expected_value_cre_xy[0]}")
            print(f"ccre(X - Y|X) = {(cre_value + (expect_value_cre_yx[0]))/cre_value}\n")
            
            print(f"ccre(X - X|Y) = {(cre_value + (expected_value_cre_xy[0]))/cre_value}\n")
            print(f"ccre(Y - Y|X) = {(cre_y_value + (expect_value_cre_yx[0]))/cre_y_value}\n")
            print(f"ccre(Y - X|Y) = {(cre_y_value + (expected_value_cre_xy[0]))/cre_y_value}\n")
            plt.scatter(new_x, new_y)
            plt.show()
if __name__ == '__main__':
    unittest.main()