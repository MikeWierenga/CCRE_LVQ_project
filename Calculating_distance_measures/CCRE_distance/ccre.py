# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.html#scipy.stats.rv_continuous
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.expect.html#scipy.stats.rv_continuous.expect
import numpy as np
from scipy import stats
from scipy.stats.contingency import margins
class CCRE:
    def __init__(self, mu, cov, invcov):
        pass
    
    def calculate_covariance_matrix(self, data):
        return np.cov(data.astype(float), rowvar=False)

    def calculate_joint_dist(self, data):
        """
        based on the formula given on this website(05/10/2023): https://www.geeksforgeeks.org/visualizing-the-bivariate-gaussian-distribution-in-python/
        returns: nx1 joint probability distribution function
        """
        # split this function so that we keep the mu cov and invcov
        cov = self.calculate_covariance_matrix(data)
        invcov = np.linalg.pinv(cov)
        
        x = data[:, 0]
        y = data[:, 1]
        mu = np.mean(np.array([x,y ]), axis=1)
        data = data.T
        long_version = []
        for i in range(data.shape[1]):
            first_part = (data[:, i] - mu).reshape(-1,1).T
            last_part = first_part.T
            formula =  -(first_part@invcov@last_part)/2
            final_formula = (1/ (np.sqrt(2*np.pi*np.linalg.det(cov)))) * np.exp(formula.astype(float))
            long_version.append(float(final_formula))
        return long_version 

    def calculate_margin_pdf(self, X, mu , sigma, sigma2):

        formula = (1 / (np.sqrt(2*np.pi)* sigma)) * np.exp(-((X- mu)**2) / (2*sigma2)) 
        return formula
     

    def calculate_expactation_value(self):
        
        pass
        
    def calculate_CCRE(self, entropy_X, expectation_value_YX):
        ccre = entropy_X - expectation_value_YX
        return ccre
