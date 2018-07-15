#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import xlrd 
from openpyxl import Workbook, load_workbook
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")
path='/home/lbj/Desktop/financial/'

allData = pd.read_csv(path+'all_data.csv',encoding='utf8')
allData.REVENUE=allData.REVENUE/10000
allData.T_COGS=allData.T_COGS/10000

allData=allData[allData.REVENUE!=0]

def funo(x):
    return x[0]*4+x[1]
t = allData[['END_DATE','REPORT_TYPE']]
t['END_DATE']=t['END_DATE']-min(t['END_DATE'])
allData['timelist']=t.apply(lambda x:funo(x),axis=1)

def funi(x):
    for i in range(len(x),0,-1):
        if i ==1:continue
        else:x.iloc[i-1]=x.iloc[i-1]-x.iloc[i-2]
    return x
allData['seasonRevenue']=allData.groupby(['TICKER_SYMBOL','END_DATE']).REVENUE.transform(lambda x:funi(x))

t = allData[['TICKER_SYMBOL','timelist','seasonRevenue']].rename(columns={'seasonRevenue':'lastseasonRevenue'})
t['timelist']=t['timelist']+1
allData=allData.merge(t,'left',on=['TICKER_SYMBOL','timelist'])
allData['errorRevenue']=allData['seasonRevenue']-allData['lastseasonRevenue']
del allData['timelist']
######################
dct = {'XSHE':0,'XSHG':1}
allData.EXCHANGE_CD = allData.EXCHANGE_CD.map(dct)
del allData['PUBLISH_DATE']
allData=allData.dropna(subset=['REVENUE'])
marketData = pd.read_excel(path+'Market Data_20180613.xlsx',index=False,encoding='utf8')[['TICKER_SYMBOL','END_DATE_','MARKET_VALUE']]
marketData['MARKET_VALUE']=marketData['MARKET_VALUE']/100000000

#trainValS = marketData.loc[marketData.END_DATE_=='2016-05-31']
#testS = marketData.loc[marketData.END_DATE_=='2017-05-31']
#del trainValS['END_DATE_']
#del testS['END_DATE_']
dayt=['2006-05-31','2007-05-31','2008-05-31','2009-05-31','2010-05-31','2011-05-31','2012-05-31'\
     ,'2013-05-31','2014-05-31','2015-05-31','2016-05-31','2017-05-31']
trainValS = marketData.loc[marketData.END_DATE_.isin(dayt)]

trainValS['END_DATE']=trainValS['END_DATE_'].map(lambda x:int(str(x).split('-')[0]))
del trainValS['END_DATE_']

allData0531 =allData[allData.REPORT_TYPE==2][['TICKER_SYMBOL','END_DATE','REVENUE']]
trainValS = allData0531.merge(trainValS,'left',on=['TICKER_SYMBOL','END_DATE'])
ttrainValS =trainValS.groupby('TICKER_SYMBOL').transform(lambda x:x.fillna(x.mean()))
trainValS['MARKET_VALUE']=ttrainValS['MARKET_VALUE']
trainValS = trainValS.set_index('REVENUE')

trainValS.fillna(0,inplace=True)

def score(y_true, y_pred): 
    Ej=0
    for i in range(len(y_true)):
        Ej=Ej+min(abs(y_pred[i]/y_true[i]-1),0.8)*np.log2(max(trainValS.loc[y_true[i],'MARKET_VALUE'],2))
    z = Ej/len(y_true)
    return ('cur_eval',z,1)

def lgbCV(train, test):
    col = [c for c in allData.columns if c not in ['TICKER_SYMBOL','REVENUE']]
    X = train[col]
    y = train['REVENUE'].values
    X_tes = test[col]
    y_tes = test['REVENUE'].values
    print('Training LGBM model...')
    lgb0 = lgb.LGBMRegressor(
        num_leaves=35,   #原为35
        depth=7,         #原来为8
        learning_rate=0.05,
        seed=2018,
        colsample_bytree=0.8,
        # min_child_samples=8,
        subsample=0.9,
        n_estimators=20000)
#    lgb_model = lgb0.fit(X, y, eval_set=[(X_tes, y_tes)],eval_metric=score,eval_names='cur_eval',\
    lgb_model = lgb0.fit(X, y, eval_set=[(X_tes, y_tes)],early_stopping_rounds=50)
    best_iter = lgb_model.best_iteration_
    predictors = [i for i in X.columns]
    feat_imp = pd.Series(lgb_model.feature_importances_, predictors).sort_values(ascending=False)
    feat_imp = feat_imp.to_frame()
    feat_imp.to_csv('/home/lbj/Desktop/feat_imp0417.csv')
    print(feat_imp)
    print(feat_imp.shape)
    pred = lgb_model.predict(test[col])
    test['pred'] = pred
    t=test[['TICKER_SYMBOL','END_DATE','REVENUE','pred','REVENUE']]
    t.to_csv(path+'error.csv',index=False)
    print('误差 ', mean_squared_error(test['REVENUE'], test['pred']))
    return best_iter

def sub(train, test, best_iter):
#    train = train.merge(S,'left',on='TICKER_SYMBOL')
    col = [c for c in allData.columns if c not in ['TICKER_SYMBOL','REVENUE']]
    X = train[col]
    y = train['REVENUE'].values
    print('Training LGBM model...')
    lgb0 = lgb.LGBMRegressor(
            boosting_type='dart',
        num_leaves=35,
        depth=7,
        learning_rate=0.05,
        seed=2018,
        colsample_bytree=0.8,
        # min_child_samples=8,
        subsample=0.9,
        n_estimators=best_iter)
    lgb_model = lgb0.fit(X, y)
    predictors = [i for i in X.columns]
    feat_imp = pd.Series(lgb_model.feature_importances_, predictors).sort_values(ascending=False)
    print(feat_imp)
    print(feat_imp.shape)
    # pred= lgb_model.predict(test[col])
    pred = lgb_model.predict(test[col])



a = [t for t in allData.columns if t not in ['TICKER_SYMBOL','REPORT_TYPE','END_DATE','industy','REVENUE','lastseasonRevenue','seasonRevenue','errorRevenue']]
for factor in a:
    allData[factor]=allData[factor].fillna(allData[factor].mean())
print('first',allData.shape)
allData.dropna(inplace=True)
print('second',allData.shape)
z=MinMaxScaler()
z.fit(allData[a])
y=z.transform(allData[a])
for i,factor in enumerate(a):
    allData[factor]=y[:,i]
for fac in ['REPORT_TYPE','END_DATE','industy']:
    dt = pd.get_dummies(allData[fac],prefix=fac)
    allData=pd.concat([allData,dt],axis=1)
print(dt.columns)
del allData['industy']
print(allData.columns) 
#train_part=allData[(allData.END_DATE<=2015)|((allData.END_DATE==2016)&(allData.REPORT_TYPE==1))]
#train_verify = allData[((allData.END_DATE==2016)&(allData.REPORT_TYPE==2))]



train = allData[(allData.END_DATE<=2016)|((allData.END_DATE==2017)&(allData.REPORT_TYPE==1))]
testdata = allData[((allData.END_DATE==2017)&(allData.REPORT_TYPE==2))]
#sub(train,testdata,best_iter)
del train['END_DATE']
del testdata['END_DATE']
del train['REPORT_TYPE']
del testdata['REPORT_TYPE']
model = LinearRegression(normalize=True)
col = [c for c in train.columns if c not in ['TICKER_SYMBOL','REVENUE']]
model.fit(train[col],train['REVENUE'])
y_pred = model.predict(testdata[col])
print(mean_squared_error(testdata['REVENUE'],y_pred))
tt=pd.DataFrame(columns=['y_pred','REVENUE'])
tt['y_pred']=y_pred
tt['REVENUE']=testdata['REVENUE'].values
tt.to_csv(path+'error.csv',index=False)
