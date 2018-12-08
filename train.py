import sys
import warnings
import argparse
import numpy as np
import pandas as pd
from data.data import process_data,cnn_data,min_day_week_data
from model import model
from keras.models import Model
warnings.filterwarnings('ignore')
def train_model(model,X_train,y_trian,name,config):
    model.compile(loss='mse',optimizer='rmsprop',metrics=['mape'])
    hist=model.fit(X_train,y_trian,
        batch_size=config['batch'],
        epochs=config['epochs'],
        validation_split=0.05)
    model.save('model/'+name+'.h5')
    df=pd.DataFrame.from_dict(hist.history)
    df.to_csv('model/'+name+'_loss.csv',encoding='utf-8',index=False)

def train_merge_lstm(model,X_train,y_trian,name,config):
    model.compile(loss='mse', optimizer='rmsprop', metrics=['mape'])
    hist = model.fit(X_train ,y_trian,
                     batch_size=config['batch'],
                     epochs=config['epochs'],
                     validation_split=0.05)
    model.save('model/' + name + '.h5')
    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv('model/' + name + '_loss.csv', encoding='utf-8', index=False)

def train_seas(models,X_train,y_train,name,config):
    temp=X_train
    for i in range(len(models)-1):
        if i>0:
            p=models[i-1]
            hidden_layer_model=Model(input=p.input,
                                     output=p.get_layer('hidden').output)
            temp=hidden_layer_model.predict(temp)
        m=models[i]
        m.compile(loss='mse',optimizer='rmsprop',metrics=['mape'])
        m.fit(temp, y_train, batch_size=config["batch"],
              epochs=config["epochs"],
              validation_split=0.05)

        models[i] = m

    saes=models[-1]
    for i in range(len(models)-1):
        weights=models[i].get_layer('hidden').get_weights()
        saes.get_layer('hidden%d'%(i+1)).set_weights(weights)
    train_model(saes,X_train,y_train,name,config)
def main(argv):
    parser=argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        default='lstm',
        help='Model to train')
    args=parser.parse_args()
    lag=12
    config={"batch": 256,"epochs":500}
    path=r'D:\data\2018_data\process_data\M30f.csv'
    if args.model in ['lstm','gru','saes']:
        X_train,y_train,_,_,_=process_data(path,lag)

    if args.model == 'merge_lstm':
        lag = [12, 96, 672]
        X_train, y_train, _, _, _ =min_day_week_data(path,lag)
        # X_train = np.reshape(X_train,[X_train.shape[0], X_train.shape[1], 1])
        m = model.merge_lstm([12,24,7,1])
        train_merge_lstm(m, X_train, y_train, args.model, config)

    if args.model=='lstm':
        X_train=np.reshape(X_train,[X_train.shape[0],X_train.shape[1],1])
        m=model.get_lstm([12,64,128,1])
        train_model(m,X_train,y_train,args.model,config)

    if args.model=='gru':
        X_train=np.reshape(X_train,[X_train.shape[0],X_train.shape[1],1])
        m=model.get_gru([12,64,128,1])
        train_model(m,X_train,y_train,args.model,config)

    if args.model=='saes':
        X_train=np.reshape(X_train,[X_train.shape[0],X_train.shape[1]])
        m=model.get_saes([12,400,400,400,1])
        train_seas(m,X_train,y_train,args.model,config)
    if args.model=='cnn':
        lag=96
        X_train,y_train,_,_,_=cnn_data(path,lag)
        X_train=np.reshape(X_train,[X_train.shape[0],X_train.shape[1],X_train.shape[2],1])
        # y_train=np.reshape(y_train,[y_train.shape[0],y_train.shape[1],1])

        m=model.get_cnn([68,96,16,32,64])
        train_model(m,X_train,y_train,args.model,config)


if __name__=='__main__':
    main(sys.argv)
