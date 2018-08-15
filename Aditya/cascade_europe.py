import numpy as np
import pandas as pd

header = ['France','Germany','Greece','Italy','Portugal','Spain']
debt_matrix = pd.read_csv('country.csv', names = header)
debt_matrix.index = header
debt_matrix = np.matrix(debt_matrix)

#      France  Germany    Greece  Italy   Portugal    Spain
gdp_08 = [2865,   3651,       351,    2307,   253,        1601]
gdp_11 = [2776,   3557,       303,    2199,   239,        1494]

gdp_08 = np.array(gdp_08)
gdp_11 = np.array(gdp_11)

normalizing_value = gdp_11[4]
gdp_08_normalized = [x / normalizing_value for x in gdp_08]
gdp_11_normalized = [x / normalizing_value for x in gdp_11]
fraction_gdp_loss = (gdp_08-gdp_11)/gdp_08

c = 0.33
n = len(header)
column_sum = np.array(sum(debt_matrix))
inverse_column_sum = 1 / column_sum

matrix_inverse_column_sum = np.matrix(np.diag(inverse_column_sum[0]))

C_matrix = c * np.dot(debt_matrix, matrix_inverse_column_sum)

C_hat = (np.array(sum((np.diag(np.ones(len(C_matrix))) - C_matrix))))[0]
C_hat = np.diag(C_hat)

A_matrix = np.dot(C_hat, np.linalg.inv((np.diag(np.ones(len(C_matrix))) - C_matrix)))

print(np.around(A_matrix,decimals=2))

theta=.9

p = np.matrix(gdp_11_normalized).T

gdp_08_normalized = np.matrix(gdp_08_normalized).T
v_threshold = theta*(np.dot(A_matrix, gdp_08_normalized))

pcurrent = p
all_failure_indicator = (np.full((1, n), False)).astype(int)
while True:
    new_failure_indicator = np.array((np.dot(A_matrix, pcurrent).T < v_threshold.T).astype(int)) * ((all_failure_indicator == 0)).astype(int)
    all_failure_indicator = all_failure_indicator | new_failure_indicator
    pcurrent = pcurrent - new_failure_indicator*v_threshold/2
    new_failed_countries = np.where(all_failure_indicator[0] == 1)[0]
    new_failed_names = header[new_failed_countries[0]]
    new_failed_names = [ header[i] for i in new_failed_countries ]

    print("Theta: ",theta)
    print(new_failed_names)
    print(new_failed_countries)
    print('\n')
    if (len(new_failed_countries)==len(header)):
        print('All countries failed for Theta: ',theta)
        break
    theta+=0.001

