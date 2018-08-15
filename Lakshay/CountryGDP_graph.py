import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

countries = ['France','Germany','Greece','Italy','Portugal','Spain']
data = pd.read_csv('normalized.csv',delimiter=',',names = countries)
data.index = countries
print(data)
#print(data['France']['Germany'])

g = nx.MultiDiGraph()

sizeofnode = []
for country in countries:
    sizeofnode.append(data[country][country])

#print(sizeofnode)
node_size = []
for country in countries:
    node_size.append(data[country][country])
    g.add_node(country, node_size = data[country][country])
    #print(country + '=')
    #print(data[country][country])

for header in countries:
    for index in countries:
        if header == index:
            pass
        else:
            #g.add_weighted_edges_from([('France','Germany',data['France']['Germany'])])
            g.add_weighted_edges_from([(header,index,data[header][index] * 100)])
        
d = nx.degree(g)
nx.draw(g, with_labels=True, node_list="header", node_size=[v * 100 for v in node_size])
plt.show()