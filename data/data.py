import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler,MinMaxScaler
path_row=r'D:\data\2018_data\raw_data'
path_process=r'D:\data\2018_data\process_data'
name_dir=['\data_6','\data_7','\data_8']
name_file=['\M30f','\M30s',r'\URBf',r'\URBs']
file=os.listdir(path_row)

def load_data(file_name):
    data=pd.read_csv(path_row+'\\'+file_name,sep=';' )
    return data


def get_flow_speed(data):
    data.columns=['id',"time","road_type","flow","ocupacion","carga","speed","error","periodo_integracion"]
    data=data[data.error=='N']

    #划分道路类型
    M30=data[data.road_type=='M30']
    URB=data[data.road_type=="URB"]

    # 交通流数据 index：time columns：路段id -1无观测值
    M30_flow=M30.pivot_table('flow',index='time',columns='id').fillna(-1)
    URB_flow=URB.pivot_table('flow',index='time',columns='id').fillna(-1)
    # 流速数据 index :time columns: 路段id  -1无观测值
    M30_speed=M30.pivot_table('speed',index='time',columns='id').fillna(-1)
    URB_speed=URB.pivot_table('speed',index='time',columns='id').fillna(-1)
    # 选取缺失值<20%的路段
    # URB_columns=URB_speed.columns[((URB_speed==-1).sum(axis=0)<1)&((URB_flow==-1).sum(axis=0)<1)]
    # M30_columns=M30_speed.columns[((M30_speed==-1).sum(axis=0)<1)&((M30_flow==-1).sum(axis=0)<1)]

    # M30_flow=M30_flow[M30_columns]
    # URB_flow=URB_flow[URB_columns]
    #
    # M30_speed=M30_speed[M30_columns]
    # URB_speed=URB_speed[URB_columns]

    return M30_flow,M30_speed,URB_flow,URB_speed



def result():

    M30f=[]
    M30s=[]
    URBf=[]
    URBs=[]
    result = [M30f,M30s,URBf,URBs]
    for i in range(3):
        data=load_data(file[i])
        Mf,Ms,Uf,Us=get_flow_speed(data)
        out=[Mf,Ms,Uf,Us]
        for j in range(4):
            result[j].append(out[j])
    return result

def to_csv():
    result1=result()
    for i in range(4):
        out_i=pd.concat(result1[i],axis=0).fillna(-1)
        out_i.to_csv(path_process+name_file[i]+'.csv',encoding='utf-8')


def process_data(path,loags):
    data=pd.read_csv(path,encoding='utf-8')
    data=data.loc[:,'3536']
    df1=data.iloc[:5856]
    df2=data.iloc[5856:]

    scaler=MinMaxScaler(feature_range=(0,1)).fit(data.values.reshape(-1,1))
    flow1=scaler.transform(df1.values.reshape(-1,1)).reshape(1,-1)[0]
    flow2=scaler.transform(df2.values.reshape(-1,1)).reshape(1,-1)[0]
    train,test=[],[]
    for i in range(loags,len(df1)):
        train.append(flow1[i-loags:i+1])
    for i in range(loags,len(df2)):
        test.append(flow2[i-loags:i+1])
    train=np.array(train)
    test=np.array(test)
    np.random.shuffle(train)

    X_train=train[:,:-1]
    y_train=train[:,-1]
    X_test=test[:,:-1]
    y_test=test[:,-1]
    return X_train,y_train,X_test,y_test,scaler

def min_day_week_data(path,lag):
    data = pd.read_csv(path, encoding='utf-8')
    data = data.loc[:, '3536']
    df1 = data.iloc[:5856]
    df2 = data.iloc[5856:]

    scaler = MinMaxScaler(feature_range=(0, 1)).fit(data.values.reshape(-1, 1))
    flow1 = scaler.transform(df1.values.reshape(-1, 1)).reshape(1, -1)[0]
    flow2 = scaler.transform(df2.values.reshape(-1, 1)).reshape(1, -1)[0]
    train_min,train_day,train_week,= [], [],[]
    test_min,test_day,test_week=[],[],[]
    day_range = np.arange(start=0, stop=lag[1],step=4)
    week_range=np.arange(start=0,stop=lag[2],step=96)

    for i in range(lag[2], len(df1)):
        train_min.append(flow1[i - lag[0]:i + 1])
    for i in range(lag[2],len(df1)):
        day=flow1[i-lag[1]:i+1]
        train_day.append(day[day_range])
    for i in range(lag[2],len(df1)):
        week=flow1[i-lag[2]:i+1]
        train_week.append(week[week_range])

    for i in range(lag[2], len(df2)):
        test_min.append(flow2[i - lag[0]:i + 1])
    for i in range(lag[2],len(df2)):
        day=flow2[i-lag[1]:i+1]
        test_day.append(day[day_range])
    for i in range(lag[2],len(df2)):
        week=flow2[i-lag[2]:i+1]
        test_week.append(week[week_range])

    train_min=np.array(train_min)
    test_min=np.array(test_min)
    train_len=train_min.shape[0]
    test_len=test_min.shape[0]
    y_train=train_min[:,-1]
    y_test=test_min[:,-1]
    train_min=train_min[:,:-1].reshape([train_len,12,1])
    test_min=test_min[:,:-1].reshape([test_len,12,1])

    train_day=np.array(train_day).reshape([train_len,24,1])
    train_week=np.array(train_week).reshape([train_len,7,1])

    test_day=np.array(test_day).reshape([test_len,24,1])
    test_week=np.array(test_week).reshape([test_len,7,1])

    X_train =[train_min,train_day,train_week]
    X_test=[test_min,test_day,test_week]

    return X_train,y_train,X_test,y_test,scaler

path1=r'D:\data\2018_data\process_data\M30f.csv'
lag=[12,96,672]
min_day_week_data(path1,lag)

def cnn_data(path,lag):
    #
    M30f=pd.read_csv(path)
    prop=len(M30f.index)*0.01
    keep_columns=M30f.columns[((M30f==0).sum(axis=0)<prop)&((M30f==-1).sum(axis=0)<prop)]
    M30f=M30f[keep_columns].iloc[:,1:].T
    d = '2018-8-1 00:00'
    columns=pd.date_range('2018-6-1 00:00',periods=8832,freq='15min')
    M30f.columns=columns

    scaler=MinMaxScaler(feature_range=(0,1)).fit(M30f.values.reshape(-1,1))
    df1=M30f.iloc[:,:5856]
    df2=M30f.iloc[:,5856:]

    flow1=scaler.transform(df1.values.reshape(-1,1)).reshape(df1.shape[0],-1)
    flow2=scaler.transform(df2.values.reshape(-1,1)).reshape(df1.shape[0],-1)


    train,test=[],[]
    for i in range(lag,(df1.shape[1])):
        train.append(flow1[:,i-lag:i+1])
    for i in range(lag,(df2.shape[1])):
        test.append(flow2[:,i-lag:i+1])

    train=np.array(train)
    test=np.array(test)

    X_train=train[:,:,:-1]
    y_train=train[:,:,-1]

    X_test=test[:,:,:-1]
    y_test=test[:,:,-1]

    # X_train = np.reshape(X_train, [X_train.shape[0], X_train.shape[1], X_train.shape[2], 1])
    print(X_train.shape)
    print(y_train.shape)

    return X_train,y_train,X_test,y_test,scaler



def flow_per15min(path1,path2):
    M30f=pd.read_csv(path1)
    prop = len(M30f.index) * 0.001
    keep_columns1 = M30f.columns[((M30f == -1).sum(axis=0) < prop)]
    M30f=M30f[keep_columns1]

    # for i in range(M30f.shape[0]-1):
    #     M30f.iloc[-(i+1),1:]=abs(M30f.iloc[-(i+1),1:]-M30f.iloc[-(i+2),1:])
    # M30f.iloc[0,1:]=0
    # keep_columns2 = M30f.columns[((M30f == 0).sum(axis=0) < prop*50)]
    # M30f=M30f[keep_columns2]
    M30f=M30f.iloc[:,1:]/4
    keep_columns2=M30f.columns[M30f.apply(pd.value_counts).max()<prop*10]
    M30f=M30f[keep_columns2]
    index= pd.date_range('2018-6-1 00:00', periods=8832, freq='15min')
    M30f.index=index
    M30f.to_csv(path2)


path1=r'D:\data\2018_data\process_data\M30f.csv'
path2=r'D:\data\2018_data\process_data\M30f_per15min.csv'
path3=r'D:\data\2018_data\process_data\M30f_per15min1.csv'
path4=r'D:\data\pmed_ubicacion_08_2018\M30_1_csv.csv'
path5=r'D:\data\2018_data\process_data\M30f_straight.csv'
patn_pems=r'D:\data\pems\contracted-Caltrans_text_gn_link_5min_2018_01_01.txt'
# data=pd.read_table(patn_pems,sep=',',header=None,usecols=[0,1,2,3,4,5,6,7],nrows=1000)
# print(data.head(),data.info())

def stright_M30(path1,path4,path5):

    columns=pd.read_csv(path4,sep=',',usecols=[1])
    columns=columns.values.reshape(1,-1)[0]
    c=[]
    for i in columns:
        c.append(str(i))
    c.append('time')
    M30f = pd.read_csv(path1,usecols=c)
    M30f.set_index('time',inplace=True)
    M30f.to_csv(path5)
