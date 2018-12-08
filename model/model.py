"""
Defination of NN model
"""
from keras import layers
from keras.layers import Dense, Dropout, Activation,Flatten,Concatenate,Input
from keras.layers.recurrent import LSTM, GRU
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential,Model
from keras.utils.vis_utils import plot_model

def get_lstm(units):
    """LSTM(Long Short-Term Memory)
    Build LSTM Model.

    # Arguments
        units: List(int), number of input, output and hidden units.
    # Returns
        model: Model, nn model.
    """
    model = Sequential()
    model.add(LSTM(units[1], input_shape=(units[0], 1), return_sequences=True))
    model.add(LSTM(units[2], return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Dense(units[3], activation='sigmoid'))

    return model


def get_gru(units):
    """GRU(Gated Recurrent Unit)
    Build GRU Model.

    # Arguments
        units: List(int), number of input, output and hidden units.
    # Returns
        model: Model, nn model.
    """

    model = Sequential()
    model.add(GRU(units[1], input_shape=(units[0], 1), return_sequences=True))
    model.add(GRU(units[2]))
    model.add(Dropout(0.2))
    model.add(Dense(units[3], activation='sigmoid'))

    return model


def _get_sae(inputs, hidden, output):
    """SAE(Auto-Encoders)
    Build SAE Model.

    # Arguments
        inputs: Integer, number of input units.
        hidden: Integer, number of hidden units.
        output: Integer, number of output units.
    # Returns
        model: Model, nn model.
    """

    model = Sequential()
    model.add(Dense(hidden, input_dim=inputs, name='hidden'))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.2))
    model.add(Dense(output, activation='sigmoid'))

    return model


def get_saes(layers):
    """SAEs(Stacked Auto-Encoders)
    Build SAEs Model.

    # Arguments
        layers: List(int), number of input, output and hidden units.
    # Returns
        models: List(Model), List of SAE and SAEs.
    """
    sae1 = _get_sae(layers[0], layers[1], layers[-1])
    sae2 = _get_sae(layers[1], layers[2], layers[-1])
    sae3 = _get_sae(layers[2], layers[3], layers[-1])

    saes = Sequential()
    saes.add(Dense(layers[1], input_dim=layers[0], name='hidden1'))
    saes.add(Activation('sigmoid'))
    saes.add(Dense(layers[2], name='hidden2'))
    saes.add(Activation('sigmoid'))
    saes.add(Dense(layers[3], name='hidden3'))
    saes.add(Activation('sigmoid'))
    saes.add(Dropout(0.2))
    saes.add(Dense(layers[4], activation='sigmoid'))

    models = [sae1, sae2, sae3, saes]

    return models

def get_cnn(unit):
    model=Sequential()
    model.add(Conv2D(kernel_size=(5,5),input_shape=(unit[0],unit[1],1),filters=unit[2],padding='same',activation='relu',data_format='channels_last'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(kernel_size=(5,5),filters=unit[3],padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(kernel_size=(5,5),filters=unit[4],padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(unit[0],activation='sigmoid'))

    return model



def merge_lstm(unit):


    input_min=Input(shape=(unit[0],1))
    lstm_min=LSTM(64,return_sequences=True)(input_min)
    lstm_min=LSTM(128)(lstm_min)
    lstm_min=Dropout(0.2)(lstm_min)
    lstm_min=Dense(12,activation='sigmoid')(lstm_min)

    input_day=Input(shape=(unit[1],1))
    lstm_day=LSTM(64,return_sequences=True)(input_day)
    lstm_day=LSTM(128)(lstm_day)
    lstm_day=Dropout(0.2)(lstm_day)
    lstm_day=Dense(24,activation='sigmoid')(lstm_day)

    input_week=Input(shape=(unit[2],1))
    lstm_week=LSTM(64,return_sequences=True)(input_week)
    lstm_week=LSTM(128)(lstm_week)
    lstm_week=Dropout(0.2)(lstm_week)
    lstm_week=Dense(7,activation='sigmoid')(lstm_week)

    merge=Concatenate()([lstm_min,lstm_day,lstm_week])
    merge=Dense(unit[3],activation='sigmoid')(merge)

    model=Model(inputs=[input_min,input_day,input_week],outputs=[merge])
    return model


# file='D:\works\PyProject\data_process\images\merge.png'
# plot_model(model,to_file=file,show_shapes=True)