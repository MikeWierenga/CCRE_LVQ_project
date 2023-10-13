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
        self.position_y = 0
        self.position_x = 1
        self.fit(data)


    def fit(self, data):
        y = data[0, :]
        x = data[1, :]
       
        self.mean = np.mean([y,x], axis=1)

        self.cov = np.cov(data)
        self.invcov = np.linalg.pinv(self.cov)
        self.detcov = np.linalg.det(self.cov)


    def calculate_joint_dist(self, new_entry):
        """
        based on the formula given on this website(05/10/2023): https://www.geeksforgeeks.org/visualizing-the-bivariate-gaussian-distribution-in-python/
        returns: 1x1 joint probability distribution function
        """
    
        first_part = (new_entry[:] - self.mean).reshape(-1,1).T
        last_part = first_part.T
        last_part = first_part.T
        
        formula =  -(first_part@self.invcov@last_part)/2
        final_formula = (1/ (np.sqrt(2*np.pi*self.detcov))) * np.exp(formula.astype(float))[0][0]
        
        return final_formula

    def mean_conditional_distribution(self):
        """
        Based on formula in book on statistics
        """
       
        mean_y = self.mean[self.position_y]
        mean_x = self.mean[self.position_x]
        cov_yx = self.cov[self.position_y][self.position_x]
        inv_cov_xx = self.invcov[self.position_x][self.position_x]
        if type(self.data) == list or type(self.data) == np.array:

            x = self.data[1, :]
        else:
            x = self.data
        mean_conditional = mean_y + cov_yx*inv_cov_xx*(x-mean_x) 
        return mean_conditional
    
    def cov_conditional_distribution(self):
        """
        Calculating the covariance of the conditional distribution p(Y|X)
        """
        
        
        cov_yy = self.cov[self.position_y][self.position_y]
        cov_yx = self.cov[self.position_y][self.position_x]
        invcov_xx = self.invcov[self.position_x][self.position_x]
        cov_xy = self.cov[self.position_x][self.position_y]

        cov_conditional = cov_yy - cov_yx*invcov_xx*cov_xy
        return cov_conditional

    def calculate_margin_pdf(self, X, mu , sigma, sigma2):

        formula = (1 / (np.sqrt(2*np.pi)* sigma)) * np.exp(-((X- mu)**2) / (2*sigma2)) 
        return formula
     

    def calculate_expactation_value(self, y, x, mu, sigma, sigma2, cov, cre_class):
        # expectation value E(X) = integral from -inf to inf of x * probability of x
        # our case E(cre(Y|X)) = cre(Y|X) * pX(x)
        if x < 0:
            print(x)
        self.data = x
        
        conditional_mean = self.mean_conditional_distribution()

        cre = cre_class.cumulative_distribution(y, conditional_mean, cov) #this will be the function to calculate the cre

        p = self.calculate_margin_pdf(x, mu, sigma, sigma2) # this will be the pdf function
        formula = cre * p
        return formula 
        
    def calculate_CCRE(self, entropy_X, expectation_value_YX):
        ccre = entropy_X - expectation_value_YX
        return ccre
