import pandas as pd
import numpy as np

from pyxlsb import open_workbook as open_xlsb

df = []

Years = ['2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014']
for year in Years:
    
    with open_xlsb('WIOT' + year + '_Nov16_ROW.xlsb') as wb:
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
    # DataIndustry.to_excel('DataIndustry' + year + '.xlsx')

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
    # DataCountry.to_csv('DataCountry' + year + '.csv')
