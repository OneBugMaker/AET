# 22-9-8
import tensorflow as tf

#%% MLP
def get_MLP(input_shape = (1600,1)):
    inputs1 = tf.keras.Input(shape =  (input_shape), name='inputs1')
    [length, depth] = input_shape
    
    xr = tf.keras.layers.Reshape((length,),name='reshape_to_1d')(inputs1)
    xr = tf.keras.layers.Dropout(0.2, name='Drop0')(xr)
    
    x1 = tf.keras.layers.Dense(256,  activation='relu',   name='x1')(xr)
    x1 = tf.keras.layers.Dropout(0.2, name='Drop1')(x1)
    
    x2 = tf.keras.layers.Dense(64,  activation='relu',   name='x2')(x1)
    x2 = tf.keras.layers.Dropout(0.2, name='Drop2')(x2)
    
    model = tf.keras.Model(inputs=inputs1, outputs=x2, name='MLP')
    return model

if __name__ == '__main__':
    MLP = get_MLP(input_shape = (2048,1))
    print( MLP.summary() )
    print('MLP', MLP.input_shape, MLP.output_shape )


#%% LSTM
def get_BiLSTM(input_shape = (1600,1)):
    inputs1 = tf.keras.Input(shape =  (input_shape), name='inputs1')
    [length, depth] = input_shape
    
    x0 = tf.keras.layers.Conv1D(64,32,32,padding='same', 
                                activation='relu',
                                name='Conv_0')(inputs1)

    # x1 = tf.keras.layers.Bidirectional(
    #             tf.keras.layers.LSTM(16, 
    #                                  return_sequences=False,
    #                                  name='LSTM_1'),
                
    #             name='BiLSTM_1')(x0)
    
    x1 = tf.keras.layers.LSTM(64, 
                                     return_sequences=True,
                                     name='LSTM_1')(x0)
    x1 = tf.keras.layers.Dropout(0.2, name='Drop1')(x1)
    
    x2 = tf.keras.layers.LSTM(64, 
                                     return_sequences=False,
                                     name='LSTM_2')(x1)
    x2 = tf.keras.layers.Dropout(0.2, name='Drop2')(x2)
    
    
    
    x3 = tf.keras.layers.Dense(64,  activation='relu',   name='x3')(x2)
    x3 = tf.keras.layers.Dropout(0.2, name='Drop3')(x3)
    
    
    model = tf.keras.Model(inputs=inputs1, outputs=x3, name='BiLSTM')
    return model


if __name__ == '__main__':
   
    LSTM = get_BiLSTM(input_shape = (2048,1))
    print( LSTM.summary() )
    print('LSTM', LSTM.input_shape, LSTM.output_shape )
    
#%% CNN
def get_CNN(input_shape = (2048,1)):
    inputs1 = tf.keras.Input(shape =  (input_shape), name='inputs1')
    [length, depth] = input_shape
    
    # xr = tf.keras.layers.Reshape(  (length, depth, 1),name='reshape_to_3d')(inputs1)
    
    x0 = tf.keras.layers.Conv1D(filters=64,  kernel_size=32, strides=32, 
                                padding='same',activation='relu',name='c0')(inputs1)
    
    
    x1 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, strides=1, 
                                padding='same',activation='relu',name='c1')(x0)
    x1 = tf.keras.layers.MaxPool1D( name='x1p')(x1)

    
    x2 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, strides=1, 
                                padding='same',activation='relu',name='c2')(x1)
    x2 = tf.keras.layers.MaxPool1D(name='x2p')(x2)
        
    #---------------------  CNN      ----------------------

    out = tf.keras.layers.Flatten(name='flatten')(x2)  # GlobalAveragePooling1D 
    out = tf.keras.layers.Dense(64,  activation='relu',   name='out')(out)
    
    model = tf.keras.Model(inputs=inputs1, outputs=out, name='CNN')
    return model

if __name__ == '__main__':
    
    CNN = get_CNN(input_shape = (2048,1))
    print( CNN.summary() )
    print('CNN', CNN.input_shape, CNN.output_shape )
    
  
#%%
def get_ResNet(input_shape = (2048,1)):
    inputs1 = tf.keras.Input(shape =  (input_shape), name='inputs1')
    [length, depth] = input_shape
    
    # xr = tf.keras.layers.Reshape(  (length, depth, 1),name='reshape_to_3d')(inputs1)
    x0 = tf.keras.layers.Conv1D(filters=64,  kernel_size=32, strides=32, 
                                padding='same',activation='relu',name='c0')(inputs1)
      
    x11 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, strides=1, 
                                padding='same',activation='relu',name='c11')(x0)
    x12 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, strides=1, 
                                padding='same',activation='relu',name='c12')(x11)
    x1a = tf.keras.layers.Add(name='add_1_2')([x11,x12])
    x1 = tf.keras.layers.MaxPool1D( name='x1p')(x1a)
    
    
    x21 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, strides=1, 
                                padding='same',activation='relu',name='c21')(x1)
    x22 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, strides=1, 
                                padding='same',activation='relu',name='c22')(x21)
    x2a = tf.keras.layers.Add(name='add_2_2')([x21,x22])
    x2 = tf.keras.layers.MaxPool1D( name='x2p')(x2a)
    #---------------------  ResNet      ----------------------
    out = tf.keras.layers.Flatten(name='flatten')(x2)  # GlobalAveragePooling1D 
    out = tf.keras.layers.Dense(64,  activation='relu',   name='out')(out)
    
    model = tf.keras.Model(inputs=inputs1, outputs=out, name='ResNet')
    return model

if __name__ == '__main__':

    ResNet = get_ResNet(input_shape = (2048,1))
    print( ResNet.summary()  )
    print('Res', ResNet.input_shape, ResNet.output_shape )
   
    
#%%
def get_HCM( dim_in, dim_out ):
    inputs1 = tf.keras.Input(shape =  dim_in , name='inputs1')
    # x11 = tf.keras.layers.Dense(128,activation=None,name='feat1')(inputs1)
    # x12 = tf.keras.layers.BatchNormalization(name='BN1')(x11)
    # x13 = tf.keras.layers.Activation('relu',name='a1')(x12)
    # x13 = tf.keras.layers.Dropout(0.1)(x13)
    
    x2 = tf.keras.layers.Dense(dim_out,  activation=None,   name='output')(inputs1)
    model = tf.keras.Model(inputs=inputs1, outputs= x2, name='HCM')
    return model


if __name__ == '__main__':

    HCM = get_HCM( dim_in=64, dim_out=10 )
    print(HCM.summary()  )
    print('HCM', HCM.input_shape, HCM.output_shape )


