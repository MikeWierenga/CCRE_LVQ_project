import numpy as np
import sys
sys.path.append('CCRE_distance')
import cre 
import ccre
import scipy.integrate as integrate

class CCRE:
    def calculate_cre(self, data):
        
        cre_x = cre.CRE(data)
        cre_value_x = cre_x.cre_gaussian_distribution()
        return cre_value_x

    def calculate_expectation_value_xy(self, data, mean_y, sigma):
            ccre_xy = ccre.CCRE(data)
            cov_conditional_dist = ccre_xy.cov_conditional_distribution()
            expected_value_cre_xy = integrate.dblquad(ccre_xy.calculate_expectation_value_xy, -np.inf, np.inf, 0, np.inf, args=(mean_y, sigma, cov_conditional_dist))
            return expected_value_cre_xy


    def calculate_ccre(self, data, x, y):
        
        old_cre_value = self.calculate_cre(x)
        old_cre_y_value = self.calculate_cre(y)


        if old_cre_value == old_cre_y_value:
            old_expected_value_cre_xy = [0]
        else:
            old_expected_value_cre_xy = self.calculate_expectation_value_xy(data.T, np.mean(y), np.std(y))

        return (old_cre_value + old_expected_value_cre_xy[0]) / old_cre_value
