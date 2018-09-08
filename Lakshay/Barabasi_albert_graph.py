# Barbasi-albert graph

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import collections
import scipy.cluster.vq as vq
import numpy.linalg as la # For finding eigenvalues and eigenvectors
import itertools
#from networkx.algorithms.community.centrality import girvan_newman
#from networkx.algorithms.community import girvan_newman
from networkx.algorithms.community import girvan_newman

# https://www.researchgate.net/publication/309153198_NumPy_SciPy_NetworkX_Recipes_for_Data_Science_Spectral_Clustering#pf3
# Reference paper for clustering

#G = nx.barabasi_albert_graph(50,10)
G = nx.powerlaw_cluster_graph(100, 1, 0.0)
#pos = nx.spring_layout(G)  # For B-A graph
pos = nx.spectral_layout(G) # For powerlaw cluster graph
nx.draw(G,pos)
plt.show()


# Calculate assortavity
r=nx.degree_assortativity_coefficient(G)
print ("Assortavity  :" + str(r))

# Calculate clustering
clustering = nx.average_clustering(G)
print("Clustering :" + str(clustering))

# Calculate density
density = nx.density(G)
print ("Density :" + str(density))

# Calculate degree distribution histogram
degree_sequence=sorted(nx.degree(G).values(),reverse=True) # degree sequence
degreeCount = collections.Counter(degree_sequence)
deg, cnt = zip(*degreeCount.items())
plt.bar(deg, cnt, width=0.80, color='b')

# PLot both graphs together
plt.axes([0.4, 0.4, 0.5, 0.5])
nx.draw(G,pos,node_size = 20)
plt.show()

# Applying k means clustering
# spectral clustering is all about eigen-vectors belonging to small eigenvalues.

print (" Applying clustering to graph")
A = nx.adjacency_matrix(G)
D = np.diag(np.ravel(np.sum(A,axis=1)))
L=D-A
l, U = la.eigh(L)
#for i in range(2,6):
 #   k=i
    #means, labels = vq.kmeans2(U[:,1:k], k)
    
  #  nx.draw(G,pos, node_color = labels)
   # plt.show()
    #print(k)
    
    


k = 6
comp = girvan_newman(G)
limited = itertools.takewhile(lambda c: len(c) <= k, comp)
for communities in limited:
    print (len([len(c) for c in sorted(communities,key=len,reverse=True)]))
    
    