# Thesis project

# About
Different medical centres use different devices and approaches to perform an FDG-PET scan image on
a patient. Combining the data from different medical centres is difficult as there are centre-specific
differences. Combining the data would be advantageous for researchers as it improves the
generalizability. To address this issue, this study proposes to apply cross cumulative residual entropy
(CCRE) to limit the influence of centre-specific differences between medical centres in FDG-PET scan
images. The data that has been used during this study consists of patients with Alzheimer's disease,
Parkinson's disease and healthy controls from the Movement Disorder Unit of the Clinica Universidad
de Navarra (CUN), The University Medical Centre Groningen (UMCG), and the University of Genoa
and IRCCS AOU San Martino-IST (UGOSM). Cross cumulative residual entropy is a measure of
uncertainty between two distributions that extends Shannon entropy in the continuous domain. To
reduce the centre specific differences CCRE has been applied as a similarity measurement. This
similarity measurement has been implemented in two machine learning methods which are k-nearest
neighbours (k-NN) and learning vector quantization (LVQ). CCRE has also been compared to Euclidean
distance, which is a well-known metric in machine learning and has been used in previous studies.
The results show that CCRE is equal in terms of accurately predicting the neurological disease in a
patient with Euclidean distance. Within Learning vector quantization, CCRE could have an advantage
if the gradient function can be worked out theoretically and applied in an existing framework.


# Data
The data used during this project consisted of FDG-PET scan images loaded via a YAML file.
PCA has been applied to these images.
If you do not have access to the data you can try this by creating Gaussian-distributed data in python.

# Code

The code has been supplied with comments and where possible there is an article to show what functions are based on (see CCRE.py for example).


# usage
To calculate all similarities run the main file in the Calculating_distance_measures folder this creates an adjacency matrix for both Euclidean distance as well as CCRE

To run Knn please see the data collection folder which uses the the adjacency matrix to classify the classes. This is used because calculating CCRE similarity takes a lot of time.

To run the LVQ model run the main file in the LVQ folder. This is an alternative version of LVQ. Normally a gradient function is written but this was not successful due to time limitation

# License
Distributed under the MIT License. SEE LICENSE for more information
