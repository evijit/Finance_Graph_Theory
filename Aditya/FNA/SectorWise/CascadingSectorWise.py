import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import networkx as nx
plt.rcParams['figure.figsize'] = [15, 15]

Graph_data = nx.MultiDiGraph()

def print_graph(failed_names):
    color_map = []
    for node in Graph_data:
        if node in failed_names:
            color_map.append('blue')
            #print (node)
        else :
            color_map.append('red')
    nx.draw(Graph_data, with_labels=True,node_color = color_map, node_size=[v * 1000 for v in node_sizes.values()])
    plt.show()
    
thetas = np.arange(0.8,1.001,0.005)
Years = ['2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014']

cascade_result = pd.DataFrame()

for y in range(1,len(Years)):
    Country_matrix1 = pd.read_csv('TotalSectorWiseData' +Years[y] + '.csv')
    Country_matrix2 = pd.read_csv('TotalSectorWiseData' +Years[y-1] + '.csv')
    Country_matrix1 = Country_matrix1.set_index('Unnamed: 0')
    Country_matrix2 = Country_matrix2.set_index('Unnamed: 0')
    
    Country_matrix1 = Country_matrix1.drop(Country_matrix1.index[[476]])
    Country_matrix2 = Country_matrix2.drop(Country_matrix2.index[[476]])
    del Country_matrix2['RUS(Entertainment)']
    del Country_matrix1['RUS(Entertainment)']
    n = len(Country_matrix1)
    
    Countries = list(Country_matrix1)
    
    gdp_1 = sum(np.matrix(Country_matrix1))
    gdp_2 = sum(np.matrix(Country_matrix2))
    normalizing_value = max(gdp_1)
    
    gdp_2_normalized = [x / normalizing_value for x in gdp_2]
    gdp_1_normalized = [x / normalizing_value for x in gdp_1]
    fraction_gdp_loss = (gdp_2-gdp_1)/gdp_2
    
    fraction_ownership = np.matrix(Country_matrix1)/gdp_1
    C_hat = np.matrix(np.diag(np.diag(fraction_ownership)))
    C_matrix = fraction_ownership - C_hat
    A_matrix = np.dot(C_hat, np.linalg.inv((np.diag(np.ones(len(C_matrix))) - C_matrix)))
    
    p = np.matrix(gdp_1_normalized[0]).T
    
    theta_failled = []

    for theta in thetas:
        v_threshold = theta*(np.dot(A_matrix, gdp_2_normalized[0].T))

        pcurrent = p
    
        all_failure_indicator = (np.full((1, n), False)).astype(int)

        node_size = list(np.array(np.dot(A_matrix, pcurrent)).T[0])
    
        node_sizes = {}
        for i in range(len(node_size)):
            node_sizes[Countries[i]] = node_size[i]
        node_sizes
    
        for country in Countries:
            Graph_data.add_node(country)
    
        for country in Countries:
            for country2 in Countries:
                if country != country2:
                    Graph_data.add_weighted_edges_from([(country, country2, Country_matrix1[country][country2])])
                
        wave = 0
        wave_failled = []
        while (wave == 0) or (len(new_failed_names) != 0):
            wave = wave + 1
    
            new_failure_indicator = (np.array((np.dot(A_matrix, pcurrent).T < v_threshold.T).astype(int)) * ((all_failure_indicator == 0))).T
            all_failure_indicator = all_failure_indicator | new_failure_indicator.T
            pcurrent = pcurrent - np.multiply(new_failure_indicator, v_threshold/2)
            new_failed_countries = np.where(all_failure_indicator[0] == 1)[0]
            a = np.where(new_failure_indicator.T[0] == 1)
    
            node_size = list(np.array(np.dot(A_matrix, pcurrent)).T[0])

            node_sizes = {}
            for i in range(len(node_size)):
                node_sizes[Countries[i]] = node_size[i]
    
            failed_names = []
    
            for x in list(new_failed_countries):
                failed_names.append(Countries[x])
     
            new_failed_names = []
    
            for x in list(a[0]):
                new_failed_names.append(Countries[x])
            wave_failled.append(new_failed_names)
            
        theta_failled.append(wave_failled)
            
    cascade_result[Years[y]] = theta_failled
cascade_result.index = thetas
cascade_result.to_csv('Result.csv')
