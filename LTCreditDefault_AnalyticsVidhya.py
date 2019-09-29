# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 21:13:16 2019

@author: snarayanaswamy
"""

import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None) ## Removes any restriction on default 
pd.set_option('display.max_rows', None) ## Removes any restriction on default 

df = pd.read_csv("C:/Users/snarayanaswamy/Downloads/LT Hackathon/train.csv")

type(df) ##Pandas dataframe
df.shape ##rows x columns
df.dtypes
df['CREDIT.HISTORY.LENGTH'].head()
df.isnull().sum() ## Counting number of missing values


## SELECTING A FEW IMPORTANT PREDICTORS
df1 = df[['loan_default','ltv','branch_id','supplier_id','manufacturer_id','Current_pincode_ID','Employment.Type',
'State_ID','MobileNo_Avl_Flag','Aadhar_flag','PAN_flag','VoterID_flag','Driving_flag','Passport_flag','PERFORM_CNS.SCORE',
'PRI.NO.OF.ACCTS','PRI.ACTIVE.ACCTS','PRI.OVERDUE.ACCTS','PRI.CURRENT.BALANCE','PRI.SANCTIONED.AMOUNT','PRI.DISBURSED.AMOUNT',
'SEC.NO.OF.ACCTS','SEC.ACTIVE.ACCTS','SEC.OVERDUE.ACCTS','SEC.CURRENT.BALANCE','SEC.SANCTIONED.AMOUNT','SEC.DISBURSED.AMOUNT',
'PRIMARY.INSTAL.AMT','SEC.INSTAL.AMT','NEW.ACCTS.IN.LAST.SIX.MONTHS','DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS',
'AVERAGE.ACCT.AGE','CREDIT.HISTORY.LENGTH','NO.OF_INQUIRIES']]

df1.head()
df1.dtypes

## EXPLORATORY CROSS TAB ANALYSIS
ct1 = pd.crosstab(index = df1['ltv_dummy1'],
                   columns= df1['ltv'])
ct1.to_csv("C:/Users/snarayanaswamy/Downloads/LT Hackathon/ltvdummy1.csv")

ct2 = pd.crosstab(index = df1['Employment.Type'],
                   columns= df1['loan_default'], normalize ='index') ## Using index creates %'s at a row level; use 'columns' if you want column level percentages
ct2

ct3 = pd.crosstab(index = df['State_ID'],
                   columns= df['loan_default'])
ct3


ct4 = pd.crosstab(index = df1['Current_pincode_ID'],
                   columns= df1['loan_default'])
ct4.to_csv("C:/Users/snarayanaswamy/Downloads/LT Hackathon/pincodefreq.csv")

## CREATING DUMMY VARIABLES BASED ON EXPLORATORY ANALYSIS
df1['manufacturer_dummy'] = np.where(df1['manufacturer_id']==48,1,0)
df1['pincode_dummy1'] = np.where(df1['Current_pincode_ID'].isin ([1055,1836,6971,1842,5092,5318,2698,4920,3792,983,2703,
2373,1660,1051,1309,1872,1726,2401,1796,7126,1854,1003,1707,3380,6258,1315,298,1840,1653,6161,5222,1867,26,7093,2382,1711,5242,
5735,90,999,1838,1577,1677,1650,1047,1869,4943,3356,5737,1713,95,975,300,1075,5660,1880,1888,31,2408,82,5104,5083,17,992,24,
6988,44,7125,7128,1859,931,1800,94,1313,1912,1698,3367,1430,1798,1860,1720,6976,1683,603,981,5748,3329,1835,1839,1858,1680,
977,88,1716,2407,1794,1849,3332,2376,3003,1043,272,1314,551,498,5844,990,5244,782,5096,2340,3241,1458,1514]),1,0)

df1['pincode_dummy2'] = np.where(df1['Current_pincode_ID'].isin ([6867,5,6855,953,6620,3382,6931,3821,1368,6591,5655,605,
2971,6897,3308,3773,3008,1518,5700,1977,2693,4609,6016,2994,5956,5659,2711,978,2983,727,3021,2213,6921,3727,3702,5726,6734,
6950,2944,1333,1631,2783,3434,6927,5661,2784,6553,2958,5662,2796,2959,2786,6898,1347,6766,2998,6744,5667,2597,3418,6296,6648,
2695,2379,2591,3404,6777,2946,730,2989,5781,3046,3442,2939,3017,5706,6771,5829,2940,2943,6628,1669,5994,1244,6890,2584,6925,
5708,3759,2700,2095,5696,6814,6861,2593,3000,3718,6763,6852,6559,6642,6940,6571,6017,6801,5704,6692,6883,6885,5710,5732,
6879,6701,6875,6882,2605,6874]),1,0)

df1['branchid_dummy1'] = np.where(df1['branch_id'].isin ([251,254,97,36,78,153,117,146,105,65,16,35,10,158,74,147]),1,0)

df1['branchid_dummy2'] = np.where(df1['branch_id'].isin ([34,42,135,82,15,3,66,162,142,104,19,100,1,17,8,152]),1,0)

df1['supplierid_dummy1'] = np.where(df1['supplier_id'].isin ([22994,18471,20716,23854,23362,20292,17705,20512,23645,16166,
23512,17014,18317,21251,13975,14143,16846,22993,16487,16445,15777,21911,18473,22971,17138,16803,15359,17742,18520,21883,
21980,20514,15694,17467,15097,23480,21498,18459,23452]),1,0)

df1['supplierid_dummy2'] = np.where(df1['supplier_id'].isin ([16556,18651,18166,21478,14004,23799,18823,16624,16120,23136,
15979,15217,15685,15805,16309,22732,15804,17901,21435,17315,18559,14441,22350,23355,17916,15899,18077,14237,14343,22056,20878,
18310,13948,16167,16679,15893,17431,16277,23323,14375,21773,22917,15271,14823,17904,17906,13890,14716,15733,22945,15460,13984,
22808]),1,0)

df1['stateid_dummy1'] = np.where(df1['State_ID'].isin ([13,14,2,12]),1,0)

df1['stateid_dummy2'] = np.where(df1['State_ID'].isin ([3,16,19,1,10,20,22]),1,0)

## CAPPING VARIABLES
df1['NEW.ACCTS.IN.LAST.SIX.MONTHS1'] = np.clip(df1['NEW.ACCTS.IN.LAST.SIX.MONTHS'],0,7)
df1['DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS1'] = np.clip(df1['DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS'],0,3)
df1['NO.OF_INQUIRIES1'] = np.clip(df1['NO.OF_INQUIRIES'],0,5)
df1['PRI.ACTIVE.ACCTS1'] = np.clip(df1['PRI.ACTIVE.ACCTS'],0,9)
df1['PRI.OVERDUE.ACCTS1'] = np.clip(df1['PRI.OVERDUE.ACCTS'],0,2)
df1['SEC.ACTIVE.ACCTS1'] = np.clip(df1['SEC.ACTIVE.ACCTS'],0,2)
df1['SEC.OVERDUE.ACCTS1'] = np.clip(df1['SEC.OVERDUE.ACCTS'],0,1)

## CREATING BINS OUT OF CONTINUOUS VARIABLES
df1['ltv_dummy1'] = np.where(((df1['ltv'] <=68.88)),1,0)
df1['ltv_dummy2'] = np.where(((df1['ltv']>=68.89) & (df1['ltv']<=76.8)),1,0)
df1['ltv_dummy3'] = np.where(((df1['ltv']>=76.81) & (df1['ltv']<=83.67)),1,0)
df1['ltv_dummy4'] = np.where(((df1['ltv'] >=83.68)),1,0)
df1['PRI.SANCTIONED.AMOUNT_dummy1'] = np.where((df1['PRI.SANCTIONED.AMOUNT'] >=62501),1,0)
df1['SEC.CURRENT.BALANCE_dummy1'] = np.where((df1['SEC.CURRENT.BALANCE'] >=1),1,0)
df1['SEC.INSTAL.AMT_dummy1'] = np.where((df1['SEC.INSTAL.AMT'] >=2),1,0)

## CREATING NEW PREDICTORS
df1['PRI.current_to_sanctioned'] = df1['PRI.CURRENT.BALANCE']/df1['PRI.SANCTIONED.AMOUNT']
df1['PRI.current_to_sanctioned'] = df1['PRI.current_to_sanctioned'].fillna(1) ## IF you dont assign, then it will create another dataframe

df1['PRI.current_to_disbursed'] = df1['PRI.CURRENT.BALANCE']/df1['PRI.DISBURSED.AMOUNT']
df1['PRI.current_to_disbursed'] = df1['PRI.current_to_disbursed'].fillna(1) ## IF you dont assign, then it will create another dataframe

df1['PRI.total_to_active_accts'] = df1['PRI.NO.OF.ACCTS']/df1['PRI.ACTIVE.ACCTS'].copy()
df1['PRI.total_to_active_accts'] = df1['PRI.total_to_active_accts'].fillna(1) ## IF you dont assign, then it will create another dataframe


##df1['SEC.current_to_sanctioned'] = df1['SEC.CURRENT.BALANCE']/df1['SEC.SANCTIONED.AMOUNT']
##df1['SEC.current_to_sanctioned'] = df1['SEC.current_to_sanctioned'].fillna(1) ## IF you dont assign, then it will create another dataframe

##df1['SEC.current_to_disbursed'] = df1['SEC.CURRENT.BALANCE']/df1['SEC.DISBURSED.AMOUNT']
##df1['SEC.current_to_disbursed'] = df1['SEC.current_to_disbursed'].fillna(1) ## IF you dont assign, then it will create another dataframe

x.isnull().sum()

##testing = df1[['PRI.CURRENT.BALANCE','PRI.SANCTIONED.AMOUNT','PRI.current_to_sanctioned']].copy() ## Creating a datafrome from 
                                                                                                  ## another dataframe


df2 = df1.drop(['Employment.Type','manufacturer_id','Current_pincode_ID','branch_id','supplier_id','State_ID','NEW.ACCTS.IN.LAST.SIX.MONTHS','DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS',
                'NO.OF_INQUIRIES','PRI.ACTIVE.ACCTS','PRI.OVERDUE.ACCTS','PRI.SANCTIONED.AMOUNT','PRI.DISBURSED.AMOUNT',
                'SEC.ACTIVE.ACCTS','SEC.OVERDUE.ACCTS','SEC.CURRENT.BALANCE','MobileNo_Avl_Flag','PAN_flag','Driving_flag',
                'Passport_flag','PRIMARY.INSTAL.AMT','SEC.INSTAL.AMT'],axis=1)

df2.dtypes
df2.to_csv("C:/Users/snarayanaswamy/Downloads/LT Hackathon/train_new.csv")
df2.columns.values ## You can use df2.dtypes

## MORE EXPLORATORY ANALYSIS
df2.groupby('loan_default').mean()


df3 = pd.read_csv("C:/Users/snarayanaswamy/Downloads/LT Hackathon/train_new.csv")

df


########################################################################################################################
################ SPLITTING DATASET INTO DEPENDENT VARIABLE AND OTHER PREDICTORS ########################################
########################################################################################################################

from sklearn.model_selection import train_test_split

y=df3['loan_default']
x= df3.drop(['loan_default'],axis=1)
x1=df2['PRI.total_to_active_accts']


## CREATING TRAINING AND TESTING DATASET
x_train,x_test, y_train, y_test = train_test_split(x,y,test_size=0.4) ## training dataset is split into training & testing samples


########################################################################################################################
################################## RECURSIVE FEATURE ELIMINATION  ######################################################
########################################################################################################################

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

rfe = RFE(logreg, 10000)

## Qn: FAILED TO CONVERGE. WHY???
rfe = rfe.fit(x,y) 

print(rfe.support_)
print(rfe.ranking_)


import statsmodels.api as sm
logit_model=sm.Logit(y,x)
result=logit_model.fit()
result.summary()

########################################################################################################################
############################################## TRAINING THE MODEL ######################################################
########################################################################################################################

## LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression

logregfit = LogisticRegression(C=1000000, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='warn',
          n_jobs=None, random_state=None, solver='warn',
          tol=0.0001, verbose=0, warm_start=False)
lr=logregfit.fit(x_train, y_train)



## SCORING THE MODEL ON 40% TEST DATASET
y_pred = logregfit.predict(x_test) ##np.array


## RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100)

clf = clf.fit(x_train,y_train)

## SCORING THE MODEL ON 40% TEST DATASET
y_pred = clf.predict(x_test) ##np.array


########################################################################################################################
############################################# MODEL EVALUATION #########################################################
########################################################################################################################
from sklearn import metrics

## AUC
print(metrics.roc_auc_score(y_test,y_pred)) ## 0.5 (pathetic)


import pandas as pd
import numpy as np
import pandas.core.algorithms as algos
from pandas import Series
import scipy.stats.stats as stats
import re
import traceback
import string

max_bin = 10
force_bin = 5
# define a binning function
def mono_bin(Y, X, n = max_bin):
    
    df1 = pd.DataFrame({"X": X, "Y": Y})
    justmiss = df1[['X','Y']][df1.X.isnull()]
    notmiss = df1[['X','Y']][df1.X.notnull()]
    r = 0
    while np.abs(r) < 1:
        try:
            d1 = pd.DataFrame({"X": notmiss.X, "Y": notmiss.Y, "Bucket": pd.qcut(notmiss.X, n)})
            d2 = d1.groupby('Bucket', as_index=True)
            r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
            n = n - 1 
        except Exception as e:
            n = n - 1

    if len(d2) == 1:
        n = force_bin         
        bins = algos.quantile(notmiss.X, np.linspace(0, 1, n))
        if len(np.unique(bins)) == 2:
            bins = np.insert(bins, 0, 1)
            bins[1] = bins[1]-(bins[1]/2)
        d1 = pd.DataFrame({"X": notmiss.X, "Y": notmiss.Y, "Bucket": pd.cut(notmiss.X, np.unique(bins),include_lowest=True)}) 
        d2 = d1.groupby('Bucket', as_index=True)
    
    d3 = pd.DataFrame({},index=[])
    d3["MIN_VALUE"] = d2.min().X
    d3["MAX_VALUE"] = d2.max().X
    d3["COUNT"] = d2.count().Y
    d3["EVENT"] = d2.sum().Y
    d3["NONEVENT"] = d2.count().Y - d2.sum().Y
    d3=d3.reset_index(drop=True)
    
    if len(justmiss.index) > 0:
        d4 = pd.DataFrame({'MIN_VALUE':np.nan},index=[0])
        d4["MAX_VALUE"] = np.nan
        d4["COUNT"] = justmiss.count().Y
        d4["EVENT"] = justmiss.sum().Y
        d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y
        d3 = d3.append(d4,ignore_index=True)
    
    d3["EVENT_RATE"] = d3.EVENT/d3.COUNT
    d3["NON_EVENT_RATE"] = d3.NONEVENT/d3.COUNT
    d3["DIST_EVENT"] = d3.EVENT/d3.sum().EVENT
    d3["DIST_NON_EVENT"] = d3.NONEVENT/d3.sum().NONEVENT
    d3["WOE"] = np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
    d3["IV"] = (d3.DIST_EVENT-d3.DIST_NON_EVENT)*np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
    d3["VAR_NAME"] = "VAR"
    d3 = d3[['VAR_NAME','MIN_VALUE', 'MAX_VALUE', 'COUNT', 'EVENT', 'EVENT_RATE', 'NONEVENT', 'NON_EVENT_RATE', 'DIST_EVENT','DIST_NON_EVENT','WOE', 'IV']]       
    d3 = d3.replace([np.inf, -np.inf], 0)
    d3.IV = d3.IV.sum()
    
    return(d3)

binning=mono_bin(y,x1,n=10)
binning

binning.to_csv("C:/Users/snarayanaswamy/Downloads/LT Hackathon/ltvbins.csv")


