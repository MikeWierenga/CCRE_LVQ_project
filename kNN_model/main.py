import kNN
class main:
  def ccre_distance(self, x, y):
    print(x[0], y[0])
    # Caluclating CRE
    x= np.array(x).reshape(-1,1)
    y = np.array(y).reshape(-1,1)
    cre_x = cre.CRE(x)
    cre_x_value = cre_x.cre_gaussian_distribution()
    cre_y = cre.CRE(y)
    cre_y_value = cre_y.cre_gaussian_distribution()
    if cre_x_value == cre_y_value:
        return 0
    #calculate expecation value X|Y
    mean_y = np.mean(y)
    sigma_y = np.std(y)

    new_data= np.concatenate((x, y), axis = 1)

    ccre_distance = ccre.CCRE(new_data.T)
    cov_conditional_dist = ccre_distance.cov_conditional_distribution() 
    expect_value_cre_xy = integrate.dblquad(ccre_distance.calculate_expectation_value_xy, -np.inf, np.inf, np.mean(x), np.inf, args=(mean_y, sigma_y, cov_conditional_dist))[0]
    ccre_value = (cre_x_value + (expect_value_cre_xy))/cre_x_value
    if ccre_value <= 0:
        return 1
    else:

        return 1 - ccre_value
      
  def main(self):
    #load data
    
    #model
    knn_model = kNN.KNN(1, ccre_distance)
    #fit model
    
