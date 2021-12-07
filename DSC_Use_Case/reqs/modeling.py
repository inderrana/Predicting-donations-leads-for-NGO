#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import sys
import random


# In[146]:


# Pr. Houdart functions

from pandas             import DataFrame
from pandas             import Series
from pandas             import read_csv
from numpy              import array
from numpy              import random


def roc (dataSet: DataFrame, actuals: str, probability: str) -> DataFrame:

    (fpr,tpr,threshold) = roc_curve(array(dataSet[actuals]), array(dataSet[probability]), pos_label = 1)

    returnData = DataFrame(tpr)
    returnData.columns = ["True positive rate"]
    returnData["False positive rate"] = DataFrame(fpr)

    return returnData

def lift (dataSet: DataFrame, actuals: str, probability: str, precision: int = 20) -> DataFrame:

    summary = cumulativeResponse(dataSet = dataSet, actuals = actuals, probability = probability, precision = precision)

    summary["Lift"] = summary["Cumulative response"] / Series(summary["Average response"]).max()
    summary["Base"] = summary["Average response"] / Series(summary["Average response"]).max()

    return summary[["Quantile","Lift","Base"]]

def cumulativeResponse (dataSet: DataFrame, actuals: str, probability: str, precision: int = 20) -> DataFrame:

    internalSet = equifrequentBinning (dataSet = dataSet[[actuals, probability]], byColumn = probability, into = precision)

    internalSet["Quantile"] = internalSet[probability + "_bin"] / precision
    internalSet["obs"]      = 1

    summary = internalSet[["Quantile", actuals, "obs"]].groupby(["Quantile"], as_index = False).sum().sort_values(by = "Quantile", ascending = False)

    summary["cumulativeTarget"]     = Series(summary[actuals]).cumsum(skipna = False)
    summary["cumulativeAll"]        = Series(summary["obs"]).cumsum(skipna = False)
    summary["Cumulative response"]  = summary["cumulativeTarget"] / summary["cumulativeAll"]
    summary["Average response"]     = Series(summary["cumulativeTarget"]).max() / Series(summary["cumulativeAll"]).max()

    return summary[["Quantile","Cumulative response","Average response"]]

def cumulativeGains (dataSet: DataFrame, actuals: str, probability: str, precision: int = 20) -> DataFrame:

    internalSet = equifrequentBinning (dataSet = dataSet[[actuals, probability]], byColumn = probability, into = precision)

    internalSet["Quantile"] = internalSet[probability + "_bin"] / precision
    internalSet["obs"]      = 1

    summary = internalSet[["Quantile", actuals, "obs"]].groupby(["Quantile"], as_index = False).sum().sort_values(by = "Quantile", ascending = False)

    summary["cumulativeTarget"]     = Series(summary[actuals]).cumsum(skipna = False)
    summary["cumulativeAll"]        = Series(summary["obs"]).cumsum(skipna = False)
    summary["Cumulative gains"]     = summary["cumulativeTarget"] / Series(summary["cumulativeTarget"]).max()
    summary["Base"]                 = summary["Quantile"]

    return summary[["Quantile","Cumulative gains","Base"]]

def equifrequentBinning (dataSet: DataFrame, byColumn: str, into: int) -> DataFrame:

    internalSet = dataSet

    quanitles = []

    for i in range(into):
        quanitles.append(1 / into * (i))

    quantile = internalSet.quantile(quanitles, axis = 0)[byColumn].to_dict()

    internalSet["Bin"] = 0

    for q in quantile:
        upperBound = quantile[q]
        internalSet.loc[internalSet[byColumn] >= upperBound, byColumn + "_bin"] = int(q * into +1)

    return internalSet

def partition (dataFrame : DataFrame, splitStrategy: [float]) -> [DataFrame]:

    def assignPartition (toDataFrame: DataFrame, lowerBound: float, upperBound: float, index: int) -> int:
        if toDataFrame["random"] >= lowerBound * observations and toDataFrame["random"] < upperBound * observations:
            return index
        else:
            return int(toDataFrame["Split"])

    if type(splitStrategy) != list:
        raise KeyError("Split strategy must be an array of floating point values.")
    elif sum(splitStrategy) != 1:
        raise ValueError("Split strategy must sum to 1.")
    else:
        observations = dataFrame.shape[0]
        partitions   = len(splitStrategy)

        cumulativeSplit = 0

        data = dataFrame.copy()
        data["random"] = random.permutation(observations)
        data["Split"]  = 0

    for index, split in enumerate(splitStrategy):
        lowerSplit = cumulativeSplit
        upperSplit = cumulativeSplit + split + 1
        cumulativeSplit += split
        data["Split"] = data.apply(lambda x: assignPartition(x,lowerSplit,upperSplit,index+1), axis = 1)

    partitions = []

    for i in range(len(splitStrategy)):
        partitions.append(data.loc[data["Split"] == i+1].drop(["Split","random"], axis = 1).reset_index(drop = True))

    return partitions


# In[258]:


gift_base_6169 = pd.read_csv("./Data/gift_base_6169.csv")
gift_base_7244 = pd.read_csv("./Data/gift_base_7244.csv")
gift_base_7362 = pd.read_csv("./Data/gift_base_7362.csv")


# In[259]:


gift_base_6169.drop("Unnamed: 0", axis=1,inplace=True)
gift_base_6169.head()


# In[260]:


gift_base_6169.shape


# In[261]:


gift_base_6169.describe()


# In[262]:


gift_base_6169.info()


# In[263]:


gift_base_6169.isna().sum().sum()


# In[264]:


#gift_base_6169.columns

cols_to_fill = ['C1', 'C2',
       'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13',
       'total_amount_1_month', 'total_freq_1_month', 'total_amount_2_month',
       'total_freq_2_month', 'total_amount_3_month', 'total_freq_3_month',
       'total_amount_4_month', 'total_freq_4_month', 'total_amount_5_month',
       'total_freq_5_month', 'total_amount_6_month', 'total_freq_6_month',
       'Recency_dono','total_donated_during_dv']

for col in cols_to_fill:
    gift_base_6169[col] = gift_base_6169[col].fillna(0)

for col in cols_to_fill:
    gift_base_7244[col] = gift_base_7244[col].fillna(0)    

for col in cols_to_fill:
    gift_base_7362[col] = gift_base_7362[col].fillna(0)      


# In[154]:


cols_to_fill = ['province_Brussels','province_East Flanders','province_Flemish Brabant'
        ,'province_Hainaut','province_Liege','province_Limburg'
        ,'province_Luxembourg',	'province_Missing',	'province_Namur'
        ,'province_Walloon Brabant','province_West Flanders'
        ,'region_Flanders',	'region_Missing', 'region_Wallonia'
        ,'language_FR',	'language_NL', 'total_freq_till_now'
        ,'total_amount_till_now', 'freq_out_campaign'
        ,'amount_out_campaign', 'freq_in_campaign'
        ,'amount_in_campaign','total_amount_till_last_3_years'
        ,'total_freq_till_last_3_years','frequency_donor','total_donated']


for col in cols_to_fill:
    gift_base_6169[col] = gift_base_6169[col].fillna(0)

for col in cols_to_fill:
    gift_base_7244[col] = gift_base_7244[col].fillna(0)    

#['gender','dateOfBirth'        


# In[155]:


#Fillna
gift_base_6169["gender"].fillna(random.choice(gift_base_6169['gender'][gift_base_6169['gender'].notna()]), inplace=True)
gift_base_6169["Age"].fillna(random.choice(gift_base_6169['Age'][gift_base_6169['Age'].notna()]), inplace=True)

gift_base_7244["gender"].fillna(random.choice(gift_base_7244['gender'][gift_base_7244['gender'].notna()]), inplace=True)
gift_base_7244["Age"].fillna(random.choice(gift_base_7244['Age'][gift_base_7244['Age'].notna()]), inplace=True)


# In[156]:


cols_to_fill = ['donorID', 'campaignID', 'frequency_donor', 'total_donated', 'C1', 'C2',
       'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13',
       'total_amount_1_month', 'total_freq_1_month', 'total_amount_2_month',
       'total_freq_2_month', 'total_amount_3_month', 'total_freq_3_month',
       'total_amount_4_month', 'total_freq_4_month', 'total_amount_5_month',
       'total_freq_5_month', 'total_amount_6_month', 'total_freq_6_month',
       'Recency_dono', 'total_donated_during_dv', 'gender', 'dateOfBirth',
       'province_Brussels', 'province_East Flanders',
       'province_Flemish Brabant', 'province_Hainaut', 'province_Liege',
       'province_Limburg', 'province_Luxembourg', 'province_Missing',
       'province_Namur', 'province_Walloon Brabant', 'province_West Flanders',
       'region_Flanders', 'region_Missing', 'region_Wallonia', 'language_FR',
       'language_NL', 'total_freq_till_now', 'total_amount_till_now',
       'freq_out_campaign', 'amount_out_campaign', 'freq_in_campaign',
       'amount_in_campaign', 'total_amount_till_last_3_years',
       'total_freq_till_last_3_years']

for col in cols_to_fill:
    gift_base_6169[col] = gift_base_6169[col].fillna(0)

for col in cols_to_fill:
    gift_base_7244[col] = gift_base_7244[col].fillna(0)    


# In[157]:


#gift_base_6169['campaignID'] = 
gift_base_6169["campaign_flag"] = np.where(gift_base_6169.campaignID.isna(),1,0)

gift_base_7244["campaign_flag"] = np.where(gift_base_7244.campaignID.isna(),1,0)


# In[158]:


gift_base_6169.drop(['dateOfBirth','campaignID','total_donated_during_dv','donorID'], axis=1, inplace=True)

gift_base_7244.drop(['dateOfBirth','campaignID','total_donated_during_dv','donorID'], axis=1, inplace=True)


# In[159]:


gift_base_6169.rename({'donated_more_than30_duringdv': 'target'}, axis=1, inplace=True)

gift_base_7244.rename({'donated_more_than30_duringdv': 'target'}, axis=1, inplace=True)

gift_base_6169.head()


# In[160]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

#for having clasification reports
from sklearn.metrics import classification_report,  roc_curve, auc, confusion_matrix, roc_auc_score

# Cross Validation Score
from sklearn.model_selection import cross_val_score

#an algotithm to normalize the numbers by dividing them to thairs std
from scipy.cluster.vq import whiten

from sklearn.metrics import r2_score, accuracy_score
from scipy.stats import spearmanr, pearsonr

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
import tensorflow as tf

#import lightgbm as lgb
import h2o
from h2o.automl import H2OAutoML

from imblearn.under_sampling import RandomUnderSampler


# In[161]:


from sklearn.utils import shuffle
gift_base_6169 = shuffle(gift_base_6169, random_state=42).reset_index(drop=True)
gift_base_6169.head()


# In[162]:


import sqldf
sqldf.run("""
select
target,
count(*)
from gift_base_6169
group by 1
""")


# In[163]:


donators = sqldf.run("""
select
*
from gift_base_6169
where target =1
""")


# In[164]:


not_donators = sqldf.run("""
select
*
from gift_base_6169
where target =0
ORDER BY RANDOM()
limit 7000
""")


# In[165]:


not_donators = sqldf.run("""
select
*
from gift_base_6169
where target =0
ORDER BY RANDOM()
limit 7000
""")


# In[166]:


test_camp = sqldf.run("""
select * from not_donators
union 
select * from donators
""")

gift_base_6169 = shuffle(test_camp)


# In[167]:


gift_base_6169 = test_camp.reset_index(drop=True)
gift_base_6169 = test_camp.drop("level_0", axis=1)
gift_base_6169 = test_camp.drop("index", axis=1)


# In[168]:


gift_base_6169


# In[169]:


gift_base_6169.drop("level_0", axis=1,inplace=True)


# In[170]:


from sklearn.utils import shuffle
donators = shuffle(donators)
not_donators = shuffle(not_donators)


# In[171]:


train = gift_base_6169
test = gift_base_7244


# In[172]:


print(train.shape)

print(test.shape)


# In[173]:


x_cols = ['frequency_donor', 'total_donated', 'C1', 'C2', 'C3', 'C4',
       'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13',
       'total_amount_1_month', 'total_freq_1_month', 'total_amount_2_month',
       'total_freq_2_month', 'total_amount_3_month', 'total_freq_3_month',
       'total_amount_4_month', 'total_freq_4_month', 'total_amount_5_month',
       'total_freq_5_month', 'total_amount_6_month', 'total_freq_6_month',
       'Recency_dono', 'gender',
       'province_Brussels', 'province_East Flanders',
       'province_Flemish Brabant', 'province_Hainaut', 'province_Liege',
       'province_Limburg', 'province_Luxembourg', 'province_Missing',
       'province_Namur', 'province_Walloon Brabant', 'province_West Flanders',
       'region_Flanders', 'region_Missing', 'region_Wallonia', 'language_FR',
       'language_NL', 'total_freq_till_now', 'total_amount_till_now',
       'freq_out_campaign', 'amount_out_campaign', 'freq_in_campaign',
       'amount_in_campaign', 'total_amount_till_last_3_years',
       'total_freq_till_last_3_years', 'Age', 'campaign_flag']

y_col = ['target']


# In[174]:


X = train[x_cols]
y = train[y_col]


# In[175]:


X_train,X_test,y_train,y_test=train_test_split(X,y.values.ravel(),test_size=0.2 ,shuffle=True)


# In[176]:


# instanciate the models
#tree         = DecisionTreeClassifier()
logistic     = LogisticRegression(max_iter = 2000)
randomForest = RandomForestClassifier(n_estimators = 700, max_depth=30)
boostedTree  = GradientBoostingClassifier()
XG           = XGBClassifier()
neuralNetR    = MLPRegressor()
#neuralNetC    = MLPClassifier()
#neighbors    = KNeighborsClassifier()

# create a dict to loop through the models later on
models = {
          "logistic"     :logistic,
          "randomForest" :randomForest,
          "boostedTree"  :boostedTree,
          "XG"           :XG

          #"neuralNetR"     :MLPRegressor,
          #"neuralNetC"     :MLPClassifier
         }


# In[177]:


# fit the models
for model in models:
    models[model].fit(X_train,y_train)
    print(f"{model} has been trained successfully")


# In[178]:


# AUC
performances = {}

#test[x_cols]
#test[y_col]

for model in models:
    predictions   = models[model].predict(test[x_cols])
    probabilities = pd.DataFrame(models[model].predict_proba(test[x_cols]))[1]
    accuracy      = accuracy_score(test[y_col],predictions)
    auc           = roc_auc_score(np.array(test[y_col]),np.array(probabilities))
    
    performances[model] = {"Accuracy":accuracy,"AUC":auc}

pd.DataFrame(performances)


# In[179]:


import shutup
shutup.please()
target='target'
lifts     = {}
responses = {}
gains     = {}
data      = pd.DataFrame(test[y_col]).copy() 

for (index,model) in enumerate(models):
    data[f"proba {model}"] = pd.DataFrame(models[model].predict_proba(test[x_cols]))[1]
    lifts[model] = lift(dataSet = data, actuals = target, probability = "proba "+str(model))
    responses[model] = cumulativeResponse(dataSet = data, actuals = target, probability = "proba "+str(model))
    gains[model] = cumulativeGains(dataSet = data, actuals = target, probability = "proba "+str(model))



for model in models:
    plt.plot(lifts[model]["Quantile"], lifts[model]["Lift"], label=model)
    plt.gca().invert_xaxis()
    plt.xlabel("quantile")
    plt.ylabel("lift")
    plt.title("Lift")
    plt.legend(loc="upper left")


# In[180]:


rf_feat = pd.DataFrame(randomForest.feature_importances_, index=X.columns, columns=["Feature importance"])
xg_feat = pd.DataFrame(XG.feature_importances_, index=X.columns, columns=["Feature importance"])
bt_feat = pd.DataFrame(boostedTree.feature_importances_, index=X.columns, columns=["Feature importance"])


# In[181]:


feat_imp = pd.concat([rf_feat,xg_feat,bt_feat], axis=1)
fi_mean = feat_imp.transpose().describe().iloc[1:2].transpose().round(4)

feat_imp = pd.concat([feat_imp,fi_mean], axis=1)
feat_imp = feat_imp.sort_values(by=['mean'], ascending=False).head(15)


# In[182]:


x_cols = feat_imp.index

y_col = ['target']


# In[188]:


# Compute the correlation matrix
corr = train[x_cols].corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Heatmap
fig = plt.figure(figsize=(14,8))
sns.heatmap(corr, mask=mask, cmap=plt.cm.bwr, center=0, annot=True, fmt='.2f', square=True)


# In[ ]:


X = train[x_cols]
y = train[y_col]

X_train,X_test,y_train,y_test=train_test_split(X,y.values.ravel(),test_size=0.2 ,shuffle=True)


# In[ ]:


# fit the models
random.seed=42
for model in models:
    models[model].fit(X_train,y_train)
    print(f"{model} has been trained successfully")


# In[ ]:


# AUC
performances = {}

#test[x_cols]
#test[y_col]

for model in models:
    predictions   = models[model].predict(X_test[x_cols])
    probabilities = pd.DataFrame(models[model].predict_proba(X_test[x_cols]))[1]
    accuracy      = accuracy_score(y_test,predictions)
    auc           = roc_auc_score(np.array(y_test),np.array(probabilities))
    
    performances[model] = {"Accuracy":accuracy,"AUC":auc}

pd.DataFrame(performances)


# In[ ]:


# AUC
performances = {}

#test[x_cols]
#test[y_col]

for model in models:
    predictions   = models[model].predict(test[x_cols])
    probabilities = pd.DataFrame(models[model].predict_proba(test[x_cols]))[1]
    accuracy      = accuracy_score(test[y_col],predictions)
    auc           = roc_auc_score(np.array(test[y_col]),np.array(probabilities))
    
    performances[model] = {"Accuracy":accuracy,"AUC":auc}

pd.DataFrame(performances)


# In[ ]:


XG.predict_proba(test[x_cols])[:,1]


# In[ ]:


import scikitplot as skplt
pred = XG.predict_proba(test[x_cols])

plt.figure(figsize=(7,7))
skplt.metrics.plot_cumulative_gain(test[y_col].values, pred)
skplt.metrics.plot_lift_curve(test[y_col].values, pred)
plt.show()


# In[ ]:


import scikitplot as skplt

target='target'
lifts     = {}
responses = {}
gains     = {}
data      = pd.DataFrame(test[y_col]).copy() 

for (index,model) in enumerate(models):
    data[f"proba {model}"] = pd.DataFrame(models[model].predict_proba(test[x_cols]))[1]
    lifts[model] = lift(dataSet = data, actuals = target, probability = "proba "+str(model))
    responses[model] = cumulativeResponse(dataSet = data, actuals = target, probability = "proba "+str(model))
    gains[model] = cumulativeGains(dataSet = data, actuals = target, probability = "proba "+str(model))


for model in models:
    pred = models[model].predict_proba(X_test[x_cols])
    plt.figure(figsize=(7,7))
    skplt.metrics.plot_cumulative_gain(y_test, pred)
    plt.title(model + " Gain Curve")
    plt.show()


# In[ ]:


target='target'
lifts     = {}
responses = {}
gains     = {}
data      = pd.DataFrame(test[y_col]).copy() 

for (index,model) in enumerate(models):
    data[f"proba {model}"] = pd.DataFrame(models[model].predict_proba(test[x_cols]))[1]
    lifts[model] = lift(dataSet = data, actuals = target, probability = "proba "+str(model))
    responses[model] = cumulativeResponse(dataSet = data, actuals = target, probability = "proba "+str(model))
    gains[model] = cumulativeGains(dataSet = data, actuals = target, probability = "proba "+str(model))


for model in models:
    pred = models[model].predict_proba(X_test[x_cols])
    plt.figure(figsize=(7,7))
    skplt.metrics.plot_lift_curve(y_test, pred)
    plt.title(model + " Lift Curve")
    plt.show()


# In[ ]:


from sklearn.metrics import plot_roc_curve

classifiers = [logistic, XG, randomForest, boostedTree]
ax = plt.gca()
for i in classifiers:
    plot_roc_curve(i, test[x_cols], test[y_col], ax=ax)


# In[ ]:


XG_Proba = XG.predict_proba(test[x_cols])[:,1]


# In[ ]:


from sklearn.metrics import roc_curve, auc


fpr, tpr, thresholds = roc_curve(test[y_col], bst_tree_proba)

#Plot ROC AUC Curve 
fig = px.area(
    x=fpr, y=tpr,
    title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
    labels=dict(x='False Positive Rate', y='True Positive Rate'),
    width=700, height=500
)
fig.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0, x1=1, y0=0, y1=1
)

fig.update_yaxes(scaleanchor="x", scaleratio=1)
fig.update_xaxes(constrain='domain')
fig.show()


# In[ ]:


import tensorflow as tf
import seaborn as sn
#pred = pred_7244["predict"]

pred = np.where(XG_Proba>0.4,1,0)

cm = tf.math.confusion_matrix(labels=test[y_col],predictions=pred)

plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[ ]:


XG_proba = XG.predict_proba(test[x_cols])
boosted_proba = boostedTree.predict_proba(test[x_cols])
logistic = logistic.predict_proba(test[x_cols])


# In[ ]:


combined = (np.array(XG_proba) + np.array(boosted_proba)) / 2


# In[ ]:


roc_auc_score(np.array(test[y_col]),combined[:,1])


# In[ ]:


combined = combined[:,1]


# In[ ]:


from sklearn.metrics import roc_curve, auc


fpr, tpr, thresholds = roc_curve(test[y_col], combined)

#Plot ROC AUC Curve 
fig = px.area(
    x=fpr, y=tpr,
    title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
    labels=dict(x='False Positive Rate', y='True Positive Rate'),
    width=700, height=500
)
fig.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0, x1=1, y0=0, y1=1
)

fig.update_yaxes(scaleanchor="x", scaleratio=1)
fig.update_xaxes(constrain='domain')
fig.show()


# In[ ]:


import tensorflow as tf
import seaborn as sn

pred = np.where(combined>0.60,1,0)

cm = tf.math.confusion_matrix(labels=test[y_col],predictions=pred)

plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[ ]:


plt.figure(figsize=(7,7))

pr =  (np.array(XG_proba) + np.array(boosted_proba)) / 2
skplt.metrics.plot_lift_curve(test[y_col], pr)
plt.title("XG + Boosted Tree Lift Curve")
plt.show()


# In[103]:


h2o.init(nthreads = -1,max_mem_size_GB = 2)


# In[104]:


train = gift_base_6169
test = gift_base_7244


# In[105]:


hx_cols = x_cols + y_col
#Create H2O frame
h_train =  h2o.H2OFrame(train)
h_test = h2o.H2OFrame(test)

y = "target"

train, test =h_train.split_frame(ratios=[0.25], seed = 42)


# In[107]:


aml = H2OAutoML(seed=42, max_models=15, max_runtime_secs_per_model=60, max_runtime_secs=360)


# In[108]:


aml.train(training_frame=h_train, y=y)


# In[109]:


lb = aml.leaderboard
lb


# In[110]:


m = h2o.get_model("GBM_grid_1_AutoML_1_20211207_83356_model_3")


# In[111]:


m.varimp_plot()


# In[112]:


var_imp = m.varimp(use_pandas=True)
var_imp


# In[123]:


imp_vars = np.array(var_imp[var_imp.scaled_importance>0.021945]["variable"])
imp_vars


# In[43]:


import shap
shap.initjs()


# ### SHAP on XG Boost

# 

# In[40]:


shap_train_X = gift_base_6169[x_cols]
shap_test_X = gift_base_7244[x_cols]


shap_train_y = gift_base_6169[y_col]
shap_test_y = gift_base_7244[y_col]

#X_train,X_test,y_train,y_test=train_test_split(X,y.values.ravel(),test_size=0.2 ,shuffle=True)


# In[44]:


# initialize explainer
explainer = shap.Explainer(XG, shap_train_X)

# compute shapley values 
shap_values = explainer.shap_values(shap_test_X)


# In[45]:


shap.summary_plot(shap_values, shap_test_X, class_names=["0", "1"])


# In[62]:


import PyALE
# adapt PyALE.ale function to incorporate classification models 
# reference: https://htmlpreview.github.io/?https://github.com/DanaJomar/PyALE/blob/master/examples/ALE%20plots%20for%20classification%20models.html
def ale(target=None, print_meanres=False, **kwargs):
    if target is not None:
        class clf():
            def __init__(self, classifier):
                self.classifier = classifier
            def predict(self, X):
                return(self.classifier.predict_proba(X)[:, target])
        clf_dummy = clf(kwargs["model"])
        kwargs["model"] = clf_dummy
    if (print_meanres & len(kwargs["feature"])==1):
        mean_response = np.mean(kwargs["model"].predict(kwargs["X"]), axis=0)
        print(f"Mean response: {mean_response:.5f}")
    return PyALE.ale(**kwargs)


# In[63]:


shap_test_X.columns


# In[72]:


# ALE plot 1D - petal length, target=1 (versicolor)
ale_petal_length = ale(
    X=shap_test_X,
    model=XG,
    feature=["amount_in_campaign"],
    include_CI=True,
    target=1,
    print_meanres=True
)

# ALE plot 1D - petal length, target=1 (versicolor)
ale_petal_length = ale(
    X=shap_test_X,
    model=XG,
    feature=["amount_in_campaign"],
    include_CI=True,
    target=1,
    print_meanres=True
)


# In[106]:


shap_train_X.columns


# In[116]:


from sklearn.inspection import PartialDependenceDisplay
# weathersituation (1=clear, 2=cloudy, 3=rain/snow)
fig, ax = plt.subplots(figsize=(12, 8))
PartialDependenceDisplay.from_estimator(estimator=XG, X=shap_train_X, features=[0, 1, 28, 29, 30, 31, 32, 33, 34], ax=ax)
fig.tight_layout(pad=2.0)


# In[135]:


from sklearn.inspection import PartialDependenceDisplay

fig, ax = plt.subplots(figsize=(12, 8))
PartialDependenceDisplay.from_estimator(estimator=XG, X=shap_train_X, features=[38, 50, 52, 51, 46, 48], ax=ax)
fig.tight_layout(pad=2.0)


# In[139]:


#38, 50, 52, 51, 46, 48
fig, ax = plt.subplots(figsize=(12, 8))
PartialDependenceDisplay.from_estimator(estimator=XG, X=shap_train_X, features=[38, 50, (52, 50), 46, (46,50)], ax=ax)
fig.tight_layout(pad=2.0)


# In[183]:


feat_imp = pd.concat([rf_feat,xg_feat,bt_feat], axis=1)
fi_mean = feat_imp.transpose().describe().iloc[1:2].transpose().round(4)

feat_imp = pd.concat([feat_imp,fi_mean], axis=1)
feat_imp = feat_imp.sort_values(by=['mean'], ascending=False).head(15)

x_cols = feat_imp.index


# In[193]:


# target 0
fig, ax = plt.subplots(figsize=(10, 4))
ice = PartialDependenceDisplay.from_estimator(estimator=XG,
                        X=X_train,
                        features=[46,1,48],
                        target=0,
                        kind="both",
                        ice_lines_kw={"color":"#808080","alpha": 0.3, "linewidth": 0.5},
                        pd_line_kw={"color": "#ffa500", "linewidth": 4, "alpha":1},
                        # centered=True, # will be added in the future
                        ax=ax)
fig.suptitle("Will not donate")
fig.tight_layout(pad=2.0)

# target 1
fig1, ax1 = plt.subplots(figsize=(10, 4))
ice = PartialDependenceDisplay.from_estimator(estimator=XG,
                        X=X_train,
                        features=[46,1,48],
                        target=1,
                        kind="both",
                        ice_lines_kw={"color":"#808080","alpha": 0.3, "linewidth": 0.5},
                        pd_line_kw={"color": "#ffa500", "linewidth": 4, "alpha":1},
                        # centered=True, # will be added in the future
                        ax=ax1)
fig1.suptitle("Will donate")
fig1.tight_layout(pad=2.0)


# In[202]:


train = gift_base_6169
test = gift_base_7244

feat_imp = pd.concat([rf_feat,xg_feat,bt_feat], axis=1)
fi_mean = feat_imp.transpose().describe().iloc[1:2].transpose().round(4)

feat_imp = pd.concat([feat_imp,fi_mean], axis=1)
feat_imp = feat_imp.sort_values(by=['mean'], ascending=False).head(15)

x_cols = feat_imp.index

y_col = ['target']

X=gift_base_6169[x_cols]
y=gift_base_6169[y_col]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=5)


# In[203]:


X_train.shape


# In[240]:


import tensorflow as tf
from tensorflow import keras
from keras import backend as K

tf.random.set_seed(123)

model = keras.Sequential([
    keras.layers.Dense(17, input_shape=(15,), activation='relu'),
    keras.layers.Dense(6, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid',
    kernel_regularizer=tf.keras.regularizers.l2(0.001)
    )
])

opt = keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[241]:


model.fit(X_train, y_train, workers=2, shuffle=42, epochs=1000, batch_size=3)


# In[242]:


#Evaluate model
scores = model.evaluate(X_test, y_test)
scores[1]*100


# In[243]:


yp = model.predict(X_test)

y_pred = []
for element in yp:
    if element > 0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)

from sklearn.metrics import confusion_matrix , classification_report

print(classification_report(y_test,y_pred))


# In[244]:


import seaborn as sn
cm = tf.math.confusion_matrix(labels=y_test,predictions=y_pred)

plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[245]:


X_7244=test[x_cols]
y_7244=test[y_col]


# In[252]:


yp = model.predict(X_7244)

y_pred = []
for element in yp:
    if element > 0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)

from sklearn.metrics import confusion_matrix , classification_report

print(classification_report(y_7244,y_pred))


# In[253]:


import seaborn as sn
cm = tf.math.confusion_matrix(labels=y_7244,predictions=y_pred)

plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[ ]:





# In[ ]:




