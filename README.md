# Product-Recommendation
While Purchasing a product company wants to recommend of a  product to a customer  that if person wants to buy a specific product can  also buy a other product  which is  relevant to the specific product.
#loading necessary packages
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules 
#reading the data
eco= pd.read_csv('C:/Users/kartik/Desktop/Data Analyst working/python/onlineretail/OnlineRetail.csv')
eco.head()
eco.dropna(subset=['UnitPrice','Quantity'],inplace=True)#remove na
eco['InvoiceNo']=eco['InvoiceNo'].astype(str)#convert Order ID into integers
eco['Description']=eco['Description'].str.strip()#removesspaces from begning and end
eco['Country'].value_counts()
#seperating transaction for Beauty and hygine
ecobasket=(eco[eco['Country']=='France']
           .groupby(['InvoiceNo','Description'])['Quantity'].sum()
           .unstack().reset_index().fillna(0).set_index('InvoiceNo'))
ecobasket.head() 
 #converting all positive values to 1 and everything else to 0
def my_encode_units(x):
    if x<=0:
        return 0
    if x>=1:
        return 1
ecobasket_sets=ecobasket .applymap(my_encode_units)
ecobasket_sets
ecobasket_sets.drop('POSTAGE', inplace = True, axis=1)
#Training Model
#Generating frequent item sets
my_frequent_itemset=apriori(ecobasket_sets,min_support=0.07,use_colnames=True )
my_frequent_itemset.head()
support	itemsets

my_rule = association_rules(my_frequent_itemset,metric="lift", min_threshold=1)
my_rule.head()

#filtering rules based on condition
my_rule[(my_rule['lift']>=3)&
        (my_rule['confidence']>=0.3)]
