from scipy.integrate import quad
import numpy as np
import scipy.stats as ss
import math
from sklearn.preprocessing import MinMaxScaler
from scipy import special
from scipy import integrate


class CRE:
    """
    Calculates the CRE (cumulative residual entropy) value based on paper "Cumulative Residual Entropy: A New Meaure of Information" by Rao et al(2004)
    """
    def __init__(self, data):
        self.data = data
     
    # CRE WITH GAUSSIAN DISTRIBUTION
    def cumulative_distribution(self, x, mu, sigma):
        """
        Calculates the cumulative distribution which is needed to calculate the cre of a variable x Rao et al(2004)
        based on this post https://stats.stackexchange.com/questions/187828/how-are-the-error-function-and-standard-normal-distribution-function-related

        params:
        x: float
        mu: mean of dataset 
        sigma: standard deviation of dataset

        returns: float
        """
        error_function = ss.norm.sf(x, mu, sigma)
        # error_function = 1 - (0.5 * (1 + math.erfc((x-mu) / (sigma*np.sqrt(2)))))
        # error_function = 1 - (0.5 * (1 + math.erf( (x-mu) / (sigma*np.sqrt(2)))))
        if error_function <0:
            error_function = 0
            
        if error_function == 0:
            log_errorfunction= 0 
        else:

            # log_errorfunction = ss.norm.logsf(x, mu,sigma)
            log_errorfunction = np.log(error_function)
        return error_function*log_errorfunction
    

    
    def cre_gaussian_distribution(self):
        """
        Calculates the cre value for a continuous random variable based on Rao's paper

        returns: positive float value
        """
        data = np.array(self.data)
        sigma = np.sqrt(np.var(data))
        mu = np.mean(data)
        
        return - 1 * integrate.quad(self.cumulative_distribution, 0, np.inf, args=(mu, sigma))[0]
