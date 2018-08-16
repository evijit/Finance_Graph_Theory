import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd
import numpy as np
import networkx as nx
plt.rcParams['figure.figsize'] = [15, 15]

Graph_data = nx.MultiDiGraph()

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

matrix_data = pd.DataFrame(A_matrix, index = header)
matrix_data.columns = header

p = np.matrix(gdp_11_normalized).T

theta=.94

gdp_08_normalized = np.matrix(gdp_08_normalized).T

v_threshold = theta*(np.dot(A_matrix, gdp_08_normalized))

pcurrent = p

n = 6
all_failure_indicator = (np.full((1, n), False)).astype(int)

node_size = list(np.array(np.dot(A_matrix, pcurrent)).T[0])

node_sizes = {}
for i in range(len(node_size)):
    node_sizes[header[i]] = node_size[i]

node_sizes = {}
for i in range(len(node_size)):
    node_sizes[header[i]] = node_size[i]

for country in header:
    Graph_data.add_node(country)

for country in header:
    for country2 in header:
        if country != country2:
            Graph_data.add_weighted_edges_from([(country, country2, matrix_data[country][country2])])

wave = 0
while (wave == 0) or (len(new_failed_names) != 0):
    wave = wave + 1
    nx.draw(Graph_data, with_labels = True,  node_size=[v * 1000 for v in node_sizes.values()])
    plt.savefig('myfig' + str(wave) + '.png')
    plt.figure()
    
    new_failure_indicator = (np.array((np.dot(A_matrix, pcurrent).T < v_threshold.T).astype(int)) * ((all_failure_indicator == 0))).T
    all_failure_indicator = all_failure_indicator | new_failure_indicator.T
    pcurrent = pcurrent - np.multiply(new_failure_indicator, v_threshold/2)
    new_failed_countries = np.where(all_failure_indicator[0] == 1)[0]
    a = np.where(new_failure_indicator.T[0] == 1)
    
    node_size = list(np.array(np.dot(A_matrix, pcurrent)).T[0])

    node_sizes = {}
    for i in range(len(node_size)):
        node_sizes[header[i]] = node_size[i]
        
    for j in new_failed_countries:
        del node_sizes[header[j]]
    
    new_failed_names = []
    for x in list(a[0]):
        Graph_data.remove_node(header[x])
        new_failed_names.append(header[x])
        
    if (len(new_failed_names) != 0):
        print('Wave', wave, 'failed countries are', new_failed_names)
