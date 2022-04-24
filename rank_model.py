#coding:utf-8
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
model = [XGBRegressor(n_estimators=200)]
model.append(SVR(kernel='rbf'))
model.append(ExtraTreesRegressor(n_estimators=200))
model.append(RandomForestRegressor(n_estimators=200))
model.append(AdaBoostRegressor(n_estimators=200))
model.append(MLPRegressor(max_iter=10000))
def average(numbers):
    return sum(numbers) / float(len(numbers))
def Monte_Carlo_test(test):
    RMSE = []
    R2 = []
    MAE = []
    for _ in range(100):
        rmse,r2,mae = test
        RMSE.append(rmse)
        R2.append(r2)
        MAE.append(mae)
    return  average(R2),average(RMSE),average(MAE)
def test(model,X,Y):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,Y,test_size=0.3)
    model = model.fit(Xtrain,Ytrain)
    Ytest_pred = model.predict(Xtest)
    rmse = np.sqrt(mean_squared_error(Ytest, Ytest_pred))
    r2 = r2_score(Ytest, Ytest_pred)
    mae = mean_absolute_error(Ytest, Ytest_pred)
    #print('rmse:', rmse)
    #print('r2:', r2)
    #print('mae:', mae)
    return rmse,r2,mae


if __name__ == '__main__':
    df=pd.read_csv(r"sisso_feature.csv")# import the train file
    df = df.sample(frac=1).reset_index(drop=True)
    X=df.iloc[:,2:]
    Y=df.iloc[:,1]
    for i in model:
        print(i,Monte_Carlo_test(test(i,X,Y)))