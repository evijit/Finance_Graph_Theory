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
column_sum = np.array(sum(debt_matrix))
inverse_column_sum = 1 / column_sum

matrix_inverse_column_sum = np.matrix(np.diag(inverse_column_sum[0]))

C_matrix = c * np.dot(debt_matrix, matrix_inverse_column_sum)

C_hat = (np.array(sum((np.diag(np.ones(len(C_matrix))) - C_matrix))))[0]
C_hat = np.diag(C_hat)

A_matrix = np.dot(C_hat, np.linalg.inv((np.diag(np.ones(len(C_matrix))) - C_matrix)))

p = np.matrix(gdp_11_normalized).T

theta=.95

gdp_08_normalized = np.matrix(gdp_08_normalized).T

v_threshold = theta*(np.dot(A_matrix, gdp_08_normalized))

pcurrent = p

n = 6
all_failure_indicator = (np.full((1, n), False)).astype(int)
wave = 0
while (wave == 0) or (len(new_failed_names) != 0):
    wave = wave + 1
    new_failure_indicator = (np.array((np.dot(A_matrix, pcurrent).T < v_threshold.T).astype(int)) * ((all_failure_indicator == 0))).T
    all_failure_indicator = all_failure_indicator | new_failure_indicator.T
    pcurrent = pcurrent - np.multiply(new_failure_indicator, v_threshold/2)
    new_failed_countries = np.where(all_failure_indicator[0] == 1)[0]
    a = np.where(new_failure_indicator.T[0] == 1)
    new_failed_names = []
    for x in list(a[0]):
        new_failed_names.append(header[x])
    if (len(new_failed_names) != 0):
        print('Wave', wave, 'failed countries are', new_failed_names)
