
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from pyxlsb import open_workbook as open_xlsb


# In[204]:


df = []

with open_xlsb('WIOT2000_Nov16_ROW.xlsb') as wb:
    with wb.get_sheet(1) as sheet:
        for row in sheet.rows():
            df.append([item.v for item in row])

df = pd.DataFrame(df[1:])

matrix = df.values

industries = list(df.iloc[2].unique())
industries.remove(None)
industries.remove('(millions of US$)')
industries.remove('Final consumption expenditure by households')
industries.remove('Final consumption expenditure by non-profit organisations serving households (NPISH)')
industries.remove('Final consumption expenditure by government')
industries.remove('Gross fixed capital formation')
industries.remove('Changes in inventories and valuables')
industries.remove('Total output')
len(industries)

countries = list(df.iloc[3].unique())
countries.remove(None)
countries.remove('TOT')
len(countries)

newMatrix = matrix[5:2469, 4:2468]

ind_con = []
for country in countries:
    for indutry in industries:
        comb = country + '(' + indutry + ')'
        ind_con.append(comb)
        
DataIndustry = pd.DataFrame(newMatrix, index = ind_con)
DataIndustry.columns = ind_con
# DataIndustry.to_excel('DataIndustry2000.xlsx')

CountryMat = []
for i in range(44):
    mat = []
    for j in range(44):
        new_mat = np.matrix(newMatrix[i*56:(i+1)*56,j*56:(j+1)*56])
        mat.append(new_mat.sum())
    CountryMat.append(mat)

CountryMat = np.matrix(CountryMat)
DataCountry = pd.DataFrame(CountryMat, index = countries)
DataCountry.columns = countries
# DataCountry.to_csv('DataCountry2000.csv')


# In[183]:


matrix = df.values


# In[184]:


industries = list(df.iloc[2].unique())
industries.remove(None)
industries.remove('(millions of US$)')
industries.remove('Final consumption expenditure by households')
industries.remove('Final consumption expenditure by non-profit organisations serving households (NPISH)')
industries.remove('Final consumption expenditure by government')
industries.remove('Gross fixed capital formation')
industries.remove('Changes in inventories and valuables')
industries.remove('Total output')
len(industries)


# In[185]:


countries = list(df.iloc[3].unique())
countries.remove(None)
countries.remove('TOT')
len(countries)


# In[186]:


newMatrix = matrix[5:2469, 4:2468]
newMatrix.shape


# In[187]:


ind_con = []
for country in countries:
    for indutry in industries:
        comb = country + '(' + indutry + ')'
        ind_con.append(comb)


# In[188]:


DataIndustry = pd.DataFrame(newMatrix, index = ind_con)
DataIndustry.columns = ind_con
DataIndustry.shape


# In[189]:


DataIndustry.to_excel('DataIndustry2010.xlsx')


# In[190]:


new_mat = np.matrix(newMatrix[0*56:(0+1)*56,0*56:(0+1)*56])
new_mat.sum()


# In[191]:


CountryMat = []
for i in range(44):
    mat = []
    for j in range(44):
        new_mat = np.matrix(newMatrix[i*56:(i+1)*56,j*56:(j+1)*56])
        mat.append(new_mat.sum())
    CountryMat.append(mat)


# In[192]:


CountryMat = np.matrix(CountryMat)
DataCountry = pd.DataFrame(CountryMat, index = countries)
DataCountry.columns = countries
DataCountry.to_csv('DataCountry2010.csv')


# # CascadeTesting

# In[3]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import networkx as nx
plt.rcParams['figure.figsize'] = [15, 15]

Graph_data = nx.MultiDiGraph()


# In[4]:


Country_matrix14 = pd.read_csv('DataCountry2014.csv')
Country_matrix12 = pd.read_csv('DataCountry2012.csv')
Country_matrix14 = Country_matrix14.set_index('Unnamed: 0')
Country_matrix12 = Country_matrix12.set_index('Unnamed: 0')
Country_matrix14.head()


# In[15]:


Countries = list(Country_matrix14)
np.where(np.array(Countries) == 'IND')
Countries[21]


# In[7]:


sum(np.matrix(Country_matrix14))


# In[10]:


sum(Country_matrix14['AUT'])


# In[37]:


gdp_14 = sum(np.matrix(Country_matrix14))
gdp_12 = sum(np.matrix(Country_matrix12))
gdp_14


# In[476]:


gdp_12


# In[26]:


normalizing_value = gdp_14[0,21]


# In[73]:


gdp_12_normalized = [x / normalizing_value for x in gdp_12]
gdp_14_normalized = [x / normalizing_value for x in gdp_14]
fraction_gdp_loss = (gdp_12-gdp_14)/gdp_12


# In[74]:


gdp_12_normalized[0]


# # Using paper 

# In[135]:


fraction_ownership = np.matrix(Country_matrix14)/gdp_14
fraction_ownership


# In[136]:


C_hat = np.matrix(np.diag(np.diag(fraction_ownership)))
C_hat


# In[137]:


C_matrix = fraction_ownership - C_hat
C_matrix


# In[138]:


A_matrix = np.dot(C_hat, np.linalg.inv((np.diag(np.ones(len(C_matrix))) - C_matrix)))
A_matrix


# In[139]:


p = np.matrix(gdp_14_normalized[0]).T

theta=.95
p


# In[140]:


A_matrix.shape


# In[141]:


v_threshold = theta*(np.dot(A_matrix, gdp_12_normalized[0].T))

pcurrent = p
v_threshold


# In[142]:


n = 44
all_failure_indicator = (np.full((1, n), False)).astype(int)

node_size = list(np.array(np.dot(A_matrix, pcurrent)).T[0])
node_size


# # CompleteCascade

# In[204]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import networkx as nx
plt.rcParams['figure.figsize'] = [15, 15]

Graph_data = nx.MultiDiGraph()

n = 44

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


# In[205]:


for year in Years:
    Country_matrix1 = pd.read_csv('DataCountry' + year + '.csv')
    Country_matrix2 = pd.read_csv('DataCountry2000.csv')
    Country_matrix1 = Country_matrix1.set_index('Unnamed: 0')
    Country_matrix2 = Country_matrix2.set_index('Unnamed: 0')
    Countries = list(Country_matrix1)
    
    gdp_1 = sum(np.matrix(Country_matrix1))
    gdp_2 = sum(np.matrix(Country_matrix2))
    normalizing_value = gdp_1[0,21]
    
    gdp_2_normalized = [x / normalizing_value for x in gdp_2]
    gdp_1_normalized = [x / normalizing_value for x in gdp_1]
    fraction_gdp_loss = (gdp_2-gdp_1)/gdp_2
    
    fraction_ownership = np.matrix(Country_matrix1)/gdp_1
    C_hat = np.matrix(np.diag(np.diag(fraction_ownership)))
    C_matrix = fraction_ownership - C_hat
    A_matrix = np.dot(C_hat, np.linalg.inv((np.diag(np.ones(len(C_matrix))) - C_matrix)))
    
    p = np.matrix(gdp_14_normalized[0]).T
    
    theta_failled = []

    for theta in thetas:
        v_threshold = theta*(np.dot(A_matrix, gdp_12_normalized[0].T))

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
                    Graph_data.add_weighted_edges_from([(country, country2, matrix_data[country][country2])])
                
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
            
    cascade_result[year] = theta_failled


# In[206]:


cascade_result


# In[143]:


node_sizes = {}
for i in range(len(node_size)):
    node_sizes[Countries[i]] = node_size[i]
node_sizes


# In[144]:


for country in Countries:
    Graph_data.add_node(country)


# In[145]:


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


# In[146]:


for country in Countries:
    for country2 in Countries:
        if country != country2:
            Graph_data.add_weighted_edges_from([(country, country2, matrix_data[country][country2])])


# In[147]:


nx.draw(Graph_data, with_labels = True,  node_size=[v * 1000 for v in node_sizes.values()])


# In[134]:


new_failure_indicator = (np.array((np.dot(A_matrix, pcurrent).T < v_threshold.T).astype(int)) * ((all_failure_indicator == 0))).T
new_failure_indicator


# In[107]:


all_failure_indicator = all_failure_indicator | new_failure_indicator.T
all_failure_indicator


# In[108]:


pcurrent = pcurrent - np.multiply(new_failure_indicator, v_threshold/2)
pcurrent


# In[109]:


new_failed_countries = np.where(all_failure_indicator[0] == 1)[0]
new_failed_countries


# In[421]:


a = np.where(new_failure_indicator.T[0] == 1)
a[0]


# In[422]:


node_size = list(np.array(np.dot(A_matrix, pcurrent)).T[0])
node_size


# In[423]:


node_sizes = {}
for i in range(len(node_size)):
    node_sizes[header[i]] = node_size[i]
node_sizes


# In[424]:


new_failed_names = []
for x in list(new_failed_countries):
#     Graph_data.remove_node(header[x])
    new_failed_names.append(header[x])


# In[425]:


new_failed_names


# In[426]:


print_graph(new_failed_names)


# In[148]:


nx.draw(Graph_data, with_labels = True,  node_size=[v * 1000 for v in node_sizes.values()])
wave = 0
while (wave == 0) or (len(new_failed_names) != 0):
    wave = wave + 1
#     plt.savefig('myfig' + str(wave) + '.png')
#     plt.show()
    
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
    print_graph(failed_names)
        
    if (len(new_failed_names) != 0):
        print('Wave', wave, 'failed countries are', new_failed_names)

