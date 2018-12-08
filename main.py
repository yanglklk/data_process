import os
import math
import warnings
import numpy as np
import pandas as pd
from data.data import process_data,cnn_data,min_day_week_data
from keras.models import load_model
from keras.utils.vis_utils import plot_model
import sklearn.metrics as metrice
import matplotlib as mpl
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

def MAPE(y_true,y_pred):
    y=[x for x in y_true if x>0]
    y_pred=[y_pred[i] for i in range(len(y_true)) if y_true[i]>0]

    num=len(y)
    sum=0

    for i in range(num):
        tep=abs(y[i]-y_pred[i])/y[i]
        sum+=tep
    mape=(sum/num)*100
    return mape

def eva_regress(y_true,y_pred):
    mape=MAPE(y_true,y_pred)
    vs=metrice.explained_variance_score(y_true,y_pred)
    mae=metrice.mean_absolute_error(y_true,y_pred)
    mse=metrice.mean_squared_error(y_true,y_pred)
    r2=metrice.r2_score(y_true,y_pred)
    print('explained_variance_score: %f'% vs)
    print("mape: %f%%"% mape)
    print("mae: %f"% mae)
    print("mse: %f"% mse)
    print("rmse: %f"% math.sqrt(mse))
    print('r2: %f'% r2)


def plot_results(y_true,y_preds,names):
    d='2018-8-1 00:00'
    x=pd.date_range(d,periods=96,freq='15min')

    fig=plt.figure()
    ax=fig.add_subplot(111)

    ax.plot(x,y_true,label="true data")
    # ax.plot(x, y_preds, label=names)
    for name,y_pred in zip(names,y_preds):
        ax.plot(x,y_pred,label=name)

    plt.legend()
    plt.grid(True)
    plt.xlabel("Time of Day")
    plt.ylabel("Flow")

    date_format=mpl.dates.DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()
    plt.show()

def main():
    lstm=load_model('model/lstm.h5')
    gru=load_model('model/gru.h5')
    saes=load_model('model/saes.h5')
    merge_lstm = load_model('model/merge_lstm.h5')
    models=[lstm,gru,merge_lstm]
    names=['LSTM','GRU','merge_lstm']

    lag=12
    path=r'D:\data\2018_data\process_data\M30f.csv'

    _,_,X_test,y_test,scaler=process_data(path,lag)
    y_test=scaler.inverse_transform(y_test.reshape(-1,1)).reshape(1,-1)[0]
    y_test=y_test[672:]
    y_pred=[]
    os.environ['path']+=os.pathsep+r'D:\document\graphviz\bin'

    for name,model in zip(names,models):

        if name=='merge_lstm':
            lag = [12, 96, 672]
            _, _, X_test, y_test, scaler = min_day_week_data(path, lag)
            y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(1, -1)[0]
            predicted=model.predict(X_test)
        else:
            X_test=np.reshape(X_test,[X_test.shape[0],X_test.shape[1],1])
            predicted = model.predict(X_test[672:])
        file='images/'+name+'.png'
        plot_model(model,to_file=file,show_shapes=True)


        predicted=scaler.inverse_transform(predicted.reshape(-1,1)).reshape(1,-1)[0]
        y_pred.append(predicted[:96])
        print(name)

        eva_regress(y_test,predicted)

    plot_results(y_test[:96],y_pred,names)

def cnn_predict():
    model=load_model('model/cnn.h5')
    name='cnn'
    lag = 96
    path = r'D:\data\2018_data\process_data\M30f_per15min1.csv'
    _,_,X_test,y_test,scaler=cnn_data(path,lag)
    shape=y_test.shape
    y_test=scaler.inverse_transform(y_test.reshape(-1,1)).reshape(shape)
    file = 'images/' + name + '.png'
    plot_model(model, to_file=file, show_shapes=True)
    predicated=model.predict(X_test.reshape([X_test.shape[0],X_test.shape[1],X_test.shape[2],1]))
    predicated=scaler.inverse_transform(predicated.reshape(-1,1)).reshape(shape)

    y_pred=predicated[:96,3]
    for i in range(10):
        print('line_%d'%i)
        y_true_i=y_test[:,i].reshape(-1,1)
        y_pred_i=predicated[:,i].reshape(-1,1)
        eva_regress(y_true_i,y_pred_i)
    plot_results(y_test[:96,3],y_pred,name)
if __name__=='__main__':
    main()
