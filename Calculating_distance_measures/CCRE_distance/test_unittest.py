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
        self.use_data = [np.array([-1.22000566,  0.45111128, -0.33724852, -0.43714635, -0.51043834, -1.37807017,
 -0.56091289,  0.76933213, -0.94135428,  1.54815972,  0.2595018,   0.87146434,
  0.06198401,  0.74659295, -1.21896738, -0.17276408, -1.11255225,  1.02960501,
  2.4797674,   0.67381005,  0.01813445, -0.36687739, -0.77562592, -1.86249175,
  0.03206438,  0.42579334,  0.47501965, -0.53059019, -0.681029,   -0.72713073,
  0.92916097, -0.3664055,   1.11031154, -0.21561573, -1.41141412, -0.98404092,
  2.39061175,  1.59043682,  0.51685942,  0.94883926, -0.33751959, -1.12627091,
 -0.13954333,  0.08081919,  0.60917639,  0.32899154,  0.58169535,  0.20372804,
  2.23686981, -1.34859636,  1.41074868, -0.76354814, -1.84791957,  0.54265616,
  1.08655134,  1.20708297, -1.0551076 ,  0.20087453, -1.54752105,  0.04702004,
  0.44589911, -0.32502413,  0.8591616 , -0.05381988,  1.13050301, -0.52792818,
  0.00316157, -0.69557317, -1.04214829,  0.84568083, -1.72174385,  0.11762371,
  0.20610696, -0.14143026,  1.81212245,  1.06117676,  0.51800456, -0.33976765,
 -1.18558733, -0.92302419, -1.4275242, -0.95696534, 1.34773846,  0.42082556,
 -0.48584798, -0.241223, -1.2403486,   0.91680314,  0.92441828, -0.04589617,
  0.12300219, -0.07218198,  0.7432211,   0.57599542, -0.03032997,  0.3430134,
 -0.35903174, -2.37330982, -0.78808779,  1.4865684, ]).reshape(-1,1),
    np.array([  2.43812316,   7.8444033,   -5.12750936,   3.89906615,  -2.68581761,
  -8.14586548,  -3.7455006,    1.55671221,   2.57242509,   8.18266954,
  -9.11085612,  -5.71871684,   4.16736189,   4.15682827,   6.77754484,
   5.04678541,  -4.74769106,   6.71256845,   4.76204497,   3.38297027,
  -3.42089474,  -0.05706809,  -1.6121368, -10.42852956,  -1.00183305,
  -4.30151915,   3.79041378, -12.70811871,  -2.24367053,   1.79814651,
  -2.08376919,   0.28846687,   5.89520684,   4.4965383,   6.29007465,
  -1.26216875,   1.98832004,  -4.58429883,  -5.36839917,   5.16712752,
  -2.97600429,  -8.63779383,   1.5397708,   4.03840538, -1.78962059,
   3.27749101,  -7.00053404,   6.08352102,   0.11360231,   9.94253862,
  -3.23220955,   0.97729166,   7.01568564,   3.68248911,  -3.58410795,
 -11.16509893,  -0.30145894,  -0.72668967,  -2.69715619,  -1.81255191,
   3.77241066,   0.86182427,  -4.80696764,   1.47849839, -10.64643417,
  -0.27716523,   1.33491335,  -4.86127196,   2.30487968,   0.20607642,
   3.68784682,  -4.43190252,   5.89381157,  -1.75619471,   7.46211964,
   1.89433936,   2.18721542,   2.34163414,   0.45151155,   4.88928544,
   7.9303315,  -9.46141207, -0.47870654,  -2.81500849,  -4.08797188,
  -1.26620253,   3.98716448,  -8.33007739,   3.19996769,   0.10605331,
  -0.18111633,  -4.14127322,   3.41921678,  -0.94758858,  -5.32802042,
  -3.38873406,   2.38072787,   0.96305402,   4.45479358,  -7.02533353,]).reshape(-1,1)]
    
    def test_cumulativedistribution(self):
        self.assertIsInstance(self.cre_class.cumulative_distribution(self.x, self.mu, self.sigma), float)

    def test_cre_gaussian_distribution(self):
        self.assertGreater(self.cre_class.cre_gaussian_distribution(), 0)

    def calculate_cre(self, data):
        cre_x = cre.CRE(data)
        cre_value_x = cre_x.cre_gaussian_distribution()
        return cre_value_x

    
    def calculate_expectation_value_xy(self, data, mean_y, sigma):
        ccre_xy = ccre.CCRE(data)
        cov_conditional_dist = ccre_xy.cov_conditional_distribution()
        expected_value_cre_xy = integrate.dblquad(ccre_xy.calculate_expectation_value_xy, -np.inf, np.inf, 0, np.inf, args=(mean_y, sigma, cov_conditional_dist))
        return expected_value_cre_xy

    def calculate_expectation_value_yx(self, data, mean_y, sigma):
        ccre_yx = ccre.CCRE(data)
        cov_conditional_dist = ccre_yx.cov_conditional_distribution()
        expect_value_cre_yx = integrate.dblquad(ccre_yx.calculate_expectation_value, -np.inf, np.inf, 0, np.inf, args=(mean_y, sigma, cov_conditional_dist))
        return expect_value_cre_yx   
    
    def test_wider_Gaus_distributions(self):    
        before = 0
        for i in np.linspace(0.1, 10, 10):

            dataset = np.random.normal(0,i,1000)
            cre_value = self.calculate_cre(dataset)
            self.assertGreater(cre_value, before)
            
            before = cre_value

    def test_unrelated_to_related_gaussian_distribution(self):

        x = np.random.normal(0,1, 100).reshape(-1,1)
        y = np.random.normal(0,1, 100).reshape(-1,1)
        original_data = np.concatenate((self.use_data[0],self.use_data[1]), axis=1)
       
        
        for i in np.linspace(0, 1, 10):
            print(f'PERCENTAGE = {i}')
            m = np.array([[1,0], [np.sin(np.pi/2*i),np.cos(np.pi/2*i)]]).T

            new_data = np.dot(original_data, m)         
            
            new_x = new_data[:, 0].reshape(-1,1)
            mean_x = np.mean(new_x)
            sigma = np.std(new_x)
           
            new_y = new_data[:, 1].reshape(-1,1)            
            mean_y = np.mean(new_y)
            sigma_y = np.std(new_y)
            new_data= np.concatenate((new_y, new_x), axis = 1)
            
            cre_value = self.calculate_cre(new_x)
            cre_y_value = self.calculate_cre(new_y)
            
            new_data_xy= np.concatenate((new_x, new_y), axis = 1)
            if cre_value == cre_y_value:
                expect_value_cre_yx= [0]
                expected_value_cre_xy = [0]
            else:
               
                expect_value_cre_yx = self.calculate_expectation_value_yx(new_data.T, mean_x, sigma)
                expected_value_cre_xy = self.calculate_expectation_value_xy(new_data_xy.T, mean_y, sigma_y)
            
            print(f"cre(X) = {cre_value} cre(Y){cre_y_value} E[e(Y|X)]{expect_value_cre_yx[0]}")
            print(f"ccre(X - Y|X) = {(cre_value + (expect_value_cre_yx[0]))/cre_value}\n")
            
            print(f"ccre(X - X|Y) = {(cre_value + (expected_value_cre_xy[0]))/cre_value}\n")
            print(f"ccre(Y - Y|X) = {(cre_y_value + (expect_value_cre_yx[0]))/cre_y_value}\n")
            print(f"ccre(Y - X|Y) = {(cre_y_value + (expected_value_cre_xy[0]))/cre_y_value}\n")

    
    def test_push_the_prototype_towards_datapoint(self):
        x = self.use_data[0]
        y = self.use_data[1]
        original_data = np.concatenate((self.use_data[0],self.use_data[1]), axis=1)
        old_cre_value = self.calculate_cre(x)
        old_cre_y_value = self.calculate_cre(y)
      
        
        if old_cre_value == old_cre_y_value:
            old_expected_value_cre_xy = [0]
        else:
            old_expected_value_cre_xy = self.calculate_expectation_value_xy(original_data.T, np.mean(y), np.std(y))
        
        #change this to how simmilar you want y to be to x the percentage parameter is your change towards the X 
        percentage = 1
        new_data = []
        for i in original_data:
            if i[0] < i[1]:
                difference = i[1] - i[0]
        
            elif i[0] > i[1]:
                difference = -(i[0] - i[1])

            else:
                print('equal')
            difference = difference * percentage
           
            new_data.append([i[0], i[1]-difference])   
        new_data = np.array(new_data)
        new_x = new_data[:, 0].reshape(-1,1)
        
        new_y = new_data[:, 1].reshape(-1,1)            
        mean_y = np.mean(new_y)
        sigma_y = np.std(new_y)
        new_data= np.concatenate((new_y, new_x), axis = 1)

        cre_value = self.calculate_cre(new_x)
        cre_y_value = self.calculate_cre(new_y)
        
        new_data_xy= np.concatenate((new_x, new_y), axis = 1)
        if cre_value == cre_y_value:
            
            expected_value_cre_xy = [0]
        else:
            
           
            expected_value_cre_xy = self.calculate_expectation_value_xy(new_data_xy.T, mean_y, sigma_y)

        print(old_cre_value, old_cre_y_value, old_expected_value_cre_xy, (old_cre_value + old_expected_value_cre_xy[0])/old_cre_value)
        print(cre_value, cre_y_value, expected_value_cre_xy, (cre_value + expected_value_cre_xy[0])/cre_value)
    
    
      
if __name__ == '__main__':
    unittest.main()