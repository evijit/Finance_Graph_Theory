import matplotlib.pyplot as plt
##%matplotlib inline
import pandas as pd
import numpy as np
import networkx as nx

Graph_data = nx.MultiDiGraph()

# http://hdr.undp.org/en/content/gdp-per-capita-2011-ppp   ## GDP 2011
header = ['France','Germany','Greece','Italy','Portugal','Spain']
matrix_data = pd.read_csv('normalized.csv', names = header)
matrix_data.index = header
matrix_data = matrix_data.T
print(matrix_data)

node_sizes = []
for country in header:
    node_sizes.append(matrix_data[country][country])
    Graph_data.add_node(country, node_size = matrix_data[country][country])

for country in header:
    for country2 in header:
        if country != country2:
            Graph_data.add_weighted_edges_from([(country, country2, matrix_data[country][country2] * 100)])

nx.draw(Graph_data, nodelist= header,  node_size=[v * 100 for v in node_sizes])
plt.show()
