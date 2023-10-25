import numpy as np

class Euclidean_Distance:

    def measure_distance(self, a, b):
        """"
        param a: list of integers
        param b: list of integers
        returns: the Euclidean distance as an integer
        """
        # param a and b have to be of the same length else the euclidean calcuation will fail
        if len(a) != len(b):
            print(f"len a {len(a)} is not equal to len b {len(b)}")
            exit()
        total = 0
   
        for i in range(len(a)):
            total += ((a[i] - b[i])**2)

        euclidean_distance = np.sqrt(total)
        return euclidean_distance[0]

    def calculate_similarity(self, distance):
        """
        Calculate the distance based similarity based on this post https://stats.stackexchange.com/questions/53068/euclidean-distance-score-and-similarity
        """
        return 1 / (1+distance)

    def calculate_average(self, distances):
        """
        param distance: list of calculated distances(integers)
        returns: average distance of all calculated distances as an integer
        """

        amount = len(distances)
        total = sum(distances)
        average = total/amount
        
        return average


