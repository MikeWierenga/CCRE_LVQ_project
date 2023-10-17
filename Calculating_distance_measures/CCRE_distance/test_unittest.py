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
        for i in np.linspace(0.1, 10, 20):

            dataset = np.random.normal(0,i,1000)
            cre_class = cre.CRE(dataset)
            cre_value = cre_class.cre_gaussian_distribution()
      
            self.assertGreater(cre_value, before)
            
            before = cre_value
            
    def test_unrelated_to_related_gaussian_distribution(self):
        x = np.random.normal(0,1, 100).reshape(-1,1)
        y = np.random.normal(0,1, 100).reshape(-1,1)
        original_data = np.concatenate((x,y), axis=1)
        # print(np.cov(original_data))
        
        for i in np.linspace(0,np.pi/2, 10):
            
            m = np.array([[1,0], [np.sin(i),np.cos(i)]]).T
    
            new_data = np.dot(original_data, m)
            new_x = new_data[:, 0].reshape(-1,1)
         
            
            mean_x = np.mean(new_x)
            sigma = np.std(new_x)
            sigma2 = np.var(new_x)

            new_y = new_data[:, 1].reshape(-1,1)
            new_data= np.concatenate((new_y, new_x), axis = 1)
            # print(new_data[0])
            cre_class = cre.CRE(new_x)
            cre_value = cre_class.cre_gaussian_distribution()
            
            ccre_distance = ccre.CCRE(new_data.T)
            cov_conditional_dist = ccre_distance.cov_conditional_distribution() 
            
            expect_value_cre_yx = integrate.dblquad(ccre_distance.calculate_expactation_value, -np.inf, np.inf, 0, np.inf, args=(mean_x, sigma, sigma2, cov_conditional_dist, cre_class))
            print(f"{cre_value} {expect_value_cre_yx} \n {m}")
                        # print(cre_value)
        pass

if __name__ == '__main__':
    unittest.main()