from scipy.integrate import quad
import numpy as np
import scipy.stats as ss
import math
from sklearn.preprocessing import MinMaxScaler
from scipy import special
from scipy import integrate

class CRE:
    def __init__(self, data):
        self.data = data
        self.nbin = len(self.data)

        self.EqualSizeBinFlag = True 
    
    def calculate_edges(self):
        if self.EqualSizeBinFlag:
            
            edges = np.histogram_bin_edges(self.data, self.nbin)
           
        else:
            prc = np.linspace(0,100, self.nbin+1)
            edges = np.percentile(self.data, prc, interpolation="midpoint")
        return edges
    
    def calculate_cre(self):
        #step 1 calculate the CDF based on matlabs histcounts function
        CP, bin = np.histogram(list(self.data), list(self.calculate_edges())) 
        hist_dist = ss.rv_histogram((CP, bin))
        CP = hist_dist.cdf(bin)[1:]

        dl = np.diff(bin) / sum(np.diff(bin))
        # -1 cause we are interested in the values greater or equal than x
        FC = 1 - CP.T
 
        logFC = []
        for i in FC:
            if i == 0 :
                logFC.append(0)
            else:

                logFC.append(np.log(i))

        if dl.T.dot(FC) == 0:
            cre = 0
        else:
         
            cre_part_one = -dl.T.dot(np.multiply(FC, logFC))
            cre_part_two = dl.T.dot(FC)
            cre = np.divide(cre_part_one, cre_part_two)

        print(f'cre = {cre}')

    # CRE WITH GAUSSIAN DISTRIBUTION
    # based on this post https://stats.stackexchange.com/questions/187828/how-are-the-error-function-and-standard-normal-distribution-function-related
    def cumulative_distribution(self, x, mu, sigma):
        error_function = 1 - (0.5 * (1 + math.erf( (x-mu) / (sigma*np.sqrt(2)))))
        if error_function <0:
            error_function = 0
            
        if error_function == 0:
            log_errorfunction= 0 
        else:

            log_errorfunction = np.log(error_function)

        return error_function*log_errorfunction
    

    
    def cre_gaussian_distribution(self):
        data = np.array(self.data)
        sigma = np.sqrt(np.var(data))
        mu = np.mean(data)
        
   
        print((- 1 * integrate.quad(self.cumulative_distribution, 0, np.inf, args=(mu, sigma))[0]))
