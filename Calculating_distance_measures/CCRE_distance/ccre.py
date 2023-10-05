# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.html#scipy.stats.rv_continuous
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.expect.html#scipy.stats.rv_continuous.expect
import numpy as np
from scipy import stats
from scipy.stats.contingency import margins
class CCRE:
    def __init__(self):
        pass
    
    def calculate_covariance_matrix(self, data):
        return np.cov(data.astype(float), rowvar=False)

    def calculate_joint_dist(self, data):
        """
        based on the formula given on this website(05/10/2023): https://www.geeksforgeeks.org/visualizing-the-bivariate-gaussian-distribution-in-python/
        returns: nx1 joint probability distribution function
        """
        cov = self.calculate_covariance_matrix(data)
        x = data[:, 0]
        y = data[:, 1]
        mu = np.mean(np.array([x,y ]), axis=1 )
        data = data.T
        long_version = []
        for i in range(data.shape[1]):
            first_part = (data[:, i] - mu).reshape(-1,1).T
            last_part = first_part.T
            formula =  -(first_part@cov@last_part)/2
            final_formula = (1/ (np.sqrt(2*np.pi*np.linalg.det(cov)))) * np.exp(formula.astype(float))
            long_version.append(float(final_formula))
        return long_version 

    def calculate_margin_pdf(self, joint_pdf):
        print(margins(joint_pdf))
        # x, y = margins(joint_pdf)
        # print(x)
        # print(y)

    def calculate_expactation_value(self, x, mu, sigma, sigma2):
        # this is to calculate the expactation value in continuous form
        # E(X) = integral(0 to np.inf) X * L(X)
        # But we have a function inside of E that is E(e(Y/X))
        # so our function should look like this
        # integral from 0 to inf of e(Y/X) * L(e(Y/X)) 
    
        
        # formula_pdf_wiki = x * (1/(sigma * np.sqrt(2* np.pi))) * np.e**(-0.5* ((x - mu)/ sigma)**2)
        # 1 / density(x) * integral 0,inf y * joint density(y,x)

        expectation_value_formula_1_variabe_pdf_rao = x *  (1/ ( np.sqrt(2*np.pi) * sigma )) * np.exp(-(((x-mu)**2)/ (2*sigma2)))
        
        return expectation_value_formula_1_variabe_pdf_rao
        
    def calculate_CCRE(self, entropy, integral_XY, pdf_XY):
        expected_value = integral_XY * pdf_XY
        ccre = entropy * expected_value
        return ccre
