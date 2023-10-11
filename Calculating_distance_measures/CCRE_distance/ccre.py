# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.html#scipy.stats.rv_continuous
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.expect.html#scipy.stats.rv_continuous.expect
import numpy as np
from scipy import stats
from scipy.stats.contingency import margins
class CCRE:
    def __init__(self, data):
        self.data = data
        self.mean = 0
        self.cov = []
        self.invcov = []
        self.detcov = []
        
        self.fit(data)
        # print(f"{self.mean} \n {self.cov} \n {self.invcov} \n hi{self.detcov}")

    def fit(self, data):
        x = data[0, :]
        y = data[1, :]
        
        self.mean = np.mean([x,y], axis=1)

        self.cov = np.cov(data)
        self.invcov = np.linalg.pinv(self.cov)
        self.detcov = np.linalg.det(self.cov)
        # print(f"{self.mean} \n cov: {self.cov} \n {self.invcov} \n {self.detcov}")

    def calculate_joint_dist(self, new_entry):
        """
        based on the formula given on this website(05/10/2023): https://www.geeksforgeeks.org/visualizing-the-bivariate-gaussian-distribution-in-python/
        returns: 1x1 joint probability distribution function
        """
        # split this function so that we keep the mu cov and invcov
        
        
        first_part = (new_entry[:] - self.mean).reshape(-1,1).T
        last_part = first_part.T
        last_part = first_part.T
        
        formula =  -(first_part@self.invcov@last_part)/2
        final_formula = (1/ (np.sqrt(2*np.pi*self.detcov))) * np.exp(formula.astype(float))[0][0]
        
        return final_formula

    def calculate_margin_pdf(self, X, mu , sigma, sigma2):

        formula = (1 / (np.sqrt(2*np.pi)* sigma)) * np.exp(-((X- mu)**2) / (2*sigma2)) 
        return formula
     

    def calculate_expactation_value(self):
        
        pass
        
    def calculate_CCRE(self, entropy_X, expectation_value_YX):
        ccre = entropy_X - expectation_value_YX
        return ccre
