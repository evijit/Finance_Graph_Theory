import networkx as nx 
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np

# Defining country header
countries = ['France','Germany','Greece','Italy','Portugal','Spain']

# Importing data for country graph
graph_data = pd.read_csv('normalized.csv',delimiter=',',names = countries)
graph_data.index = countries

def create_graph():
    node_size = []
    for country in countries:
        node_size.append(graph_data[country][country])
        g.add_node(country, node_size = graph_data[country][country])
    #print(country + '=')
    #print(data[country][country])

    for header in countries:
        for index in countries:
            if header == index:
                pass
            else:
                g.add_weighted_edges_from([(header,index,graph_data[header][index] * 100)])
    return node_size

def print_graph(failed_names):
    color_map = []
    for node in g:
        if node in failed_names:
            color_map.append('blue')
            #print (node)
        else :
            color_map.append('red')
    nx.draw(g, with_labels=True,node_color = color_map, node_list="header", node_size=[v * 100 for v in node_size])
    plt.show()


#      France  Germany    Greece  Italy   Portugal    Spain
gdp_08 = [2865,   3651,       351,    2307,   253,        1601]
gdp_11 = [2776,   3557,       303,    2199,   239,        1494]

#Taking Portugal's 2011 GDP for normalizing
normalizing_value = gdp_11[4]
gdp_08_normalized = [value / normalizing_value for value in gdp_08]
gdp_11_normalized = [value / normalizing_value for value in gdp_11]
# Fractional gdp loss
fraction_gdp_loss = np.divide(np.subtract(gdp_08,gdp_11), gdp_08)
#print (fraction_gdp_loss)

# Importing country debt data
data = pd.read_csv('country.csv',delimiter=',',names = countries)
data = np.matrix(data)
n = len(countries)

column_sum = data.sum(axis = 0)

c = 0.33
column_sum_inverse = np.array(1/column_sum)
matrix_inverse_column_sum = np.matrix(np.diag(column_sum_inverse[0]))
C_matrix = c * np.dot(data, matrix_inverse_column_sum)

C_hat = (np.array(sum((np.diag(np.ones(len(C_matrix))) - C_matrix))))[0]
C_hat = np.diag(C_hat)

A_matrix = np.dot(C_hat, np.linalg.inv((np.diag(np.ones(len(C_matrix))) - C_matrix)))
#print(np.around(A_matrix,decimals=2))

# Graph setup
g = nx.MultiDiGraph()
node_size = create_graph()

# Cascade Code
p = np.matrix(gdp_11_normalized).T
gdp_08_normalized = np.matrix(gdp_08_normalized).T

theta = 0.935
v_threshold = theta * (np.dot(A_matrix, gdp_08_normalized))
pcurrent = p
wave = 0
all_failure_indicator = np.zeros(6, dtype = int)

while (wave == 0) or (len(new_failed_names) != 0):
    wave = wave + 1
    new_failure_indicator = (np.array((np.dot(A_matrix, pcurrent).T < v_threshold.T).astype(int)) * (all_failure_indicator == 0)).T
    all_failure_indicator = all_failure_indicator | new_failure_indicator.T
    pcurrent = pcurrent - np.multiply(new_failure_indicator, v_threshold/2)

    new_failed_countries = np.where(new_failure_indicator.T[0] == 1)[0]
    #new_failed_countries = np.where(all_failure_indicator[0] == 1)[0]    #To print all failed countries in a wave
    all_failed_countries = np.where(all_failure_indicator[0] == 1)[0]
    new_failed_names = []
    new_failed_names = [ countries[i] for i in new_failed_countries]
    all_failed_names = [ countries[i] for i in all_failed_countries ]
    if (len(new_failed_names) != 0):
      print("Countries failed in wave:",wave)
      #print(new_failed_names)
      print(all_failed_names)
      
    # Printing node graph
    #print_graph(new_failed_names)
    print_graph(all_failed_names)
    if (len(new_failed_countries) == n):
      break
   