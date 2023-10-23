import numpy as np
from scipy import stats
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
        
        formula =  -(first_part@self.invcov@last_part)/2
        final_formula = (1/ (np.sqrt(2*np.pi*self.detcov))) * np.exp(formula.astype(float))[0][0]
        
        return final_formula

    def mean_conditional_distribution(self):
        """
        Based on formula in book on statistics
        returns: mean of conditional distribution (float)
        """
       
        mean_y = self.mean[self.position_y]
        
        mean_x = self.mean[self.position_x]
  
        cov_yx = self.cov[self.position_y][self.position_x]
        
        inv_cov_xx = 1/self.cov[self.position_x][self.position_x]
        if type(self.data) == list or type(self.data) == np.array:

            x = self.data[1, :]
        else:
            x = self.data
        mean_conditional = mean_y + cov_yx*inv_cov_xx*(x-mean_x) 
        return mean_conditional
    
    def cov_conditional_distribution(self):
        """
        Calculating the covariance of the conditional distribution p(Y|X) based on book on statistics
        returns: covariance of conditional distribution (float)
        """
        cov_yy = self.cov[self.position_y][self.position_y]
        cov_yx = self.cov[self.position_y][self.position_x]
       
        invcov_xx = 1/self.cov[self.position_x][self.position_x]
        cov_xy = self.cov[self.position_x][self.position_y]

        cov_conditional = cov_yy - cov_yx*invcov_xx*cov_xy
        
        return cov_conditional

    def calculate_margin_pdf(self, x, mu , sigma, sigma2):
        formula = stats.norm.pdf(x, mu, sigma)
        
        return formula
     

    def calculate_expectation_value(self, y, x, mu, sigma, sigma2, cov, cre_class):
        # expectation value E(X) = integral from -inf to inf of x * probability of x
        # our case E(cre(Y|X)) = cre(Y|X) * pX(x)
        """
        Calculates the expectation value of cre(Y|X)
        """
        self.data = x
        conditional_mean = self.mean_conditional_distribution()
        cre = cre_class.cumulative_distribution(y, conditional_mean, cov)
        
        p = self.calculate_margin_pdf(x, mu, sigma, sigma2) 
        
        formula = cre * p
        return formula 
    

    
    def calculate_expectation_value_xy(self, y, x, mu_y, sigma_y, sigma2_y, cov, cre_class):
        """
        Calculates the expectation value of cre(X|Y)
        """
        self.data = y
        conditional_mean = self.mean_conditional_distribution()
        cre = cre_class.cumulative_distribution(x, conditional_mean, cov) 
        p = self.calculate_margin_pdf(y, mu_y, sigma_y, sigma2_y) 
        
        formula = cre * p
        return formula 
      
    def calculate_CCRE(self, entropy, expectation_value):
        """
        Calculates the CCRE
        """
        ccre = (entropy - expectation_value) / entropy
        return ccre
