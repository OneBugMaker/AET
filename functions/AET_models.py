# 10-7 - 多个小核卷积代替一个大核卷积

import tensorflow as tf
import numpy as np
if __name__ == '__main__': 
    from tools import positional_encoding
else:
    from functions.tools import positional_encoding
print(tf.__version__ )


#%% 缩放点积注意力
# https://tensorflow.google.cn/tutorials/text/transformer?hl=zh_cn#%E5%88%9B%E5%BB%BA_transformer
def scaled_dot_product_attention(q, k, v):
    """
    q: [batch, sequence_len, q_embeding_len]
    k: [batch, sequence_len, k_embeding_len]
    v: [batch, sequence_len, v_embeding_len]
    序列的每个位置上（序列长度这个维度）的q向量要和k向量做点积相似度计算，
    因此：q_embeding_len  ==必须==  k_embeding_len，
    sequence_len个q向量中的 每个q向量与 sequence_len个k向量中的 每个k向量求点积相似度，
    得到，sequence_len * sequence_len  个相似度结果，
    即相似度矩阵：
    attention_weights = 【sequence_len，sequence_len】
    out  =   attention_weights * value 
         =  【sequence_len，sequence_len】*【sequence_len, v_embeding_len】
         =  【sequence_len, v_embeding_len】
    
    """
    k_embeding_len = tf.cast( tf.shape(k)[-1], dtype=tf.float32 )
    sqrt_k = tf.sqrt(k_embeding_len)
    
    matmul_qk = tf.matmul( a=q,    b=k,   transpose_b=True )
    matmul_qk = matmul_qk / sqrt_k
    

        
    attention_weights = tf.math.softmax( matmul_qk , axis = -1 )
    
    out = tf.matmul( attention_weights, v )
    return out, attention_weights
    

#%% 多头注意力
# 22-9-6
class MultiHeadAttention(tf.keras.layers.Layer):
    '''
    q: [batch, sequence_len, q_embeding_len]
    k: [batch, sequence_len, k_embeding_len]
    v: [batch, sequence_len, v_embeding_len]
    k_embeding_len太长导致复杂度很高，
    点积缩放注意力的计算量为：sq_len*qkeb_len*qkeb_len*sq_len + sq_len*sq_len*sq_len*veb_len
    sq_len^2 * qkeb_len^2 + sq_len^3 * veb_len
    多头注意力：把嵌入维度平均分成 heads_num 份，再计算 heads_num 次点积缩放注意力
    其计算量为：heads_num*[ sq_len*(qkeb_len/heads_num)*(qkeb_len/heads_num)*sq_len
                            + sq_len*sq_len*sq_len*(veb_len/heads_num)  ]
    sq_len^2 * qkeb_len^2/heads_num + sq_len^3 * veb_len        
    '''
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model   = d_model
        
        assert d_model%self.num_heads == 0
        
        self.depth = d_model//self.num_heads
        
        self.q_linear_mapping = tf.keras.layers.Dense(d_model)
        self.k_linear_mapping = tf.keras.layers.Dense(d_model)
        self.v_linear_mapping = tf.keras.layers.Dense(d_model)
        
        self.out_linear_mapping = tf.keras.layers.Dense(d_model)
        
    def split_heads(self, x, batch_size):
        """
        x.shape = 【batch_size, seq_len, embeding_len】
        （1）把每个位置的embeding向量均分为 num_heads 份，
        即把最后一个维度（代表 embeding 长度）
        变为两个维度【batch_size, seq_len, num_heads，embeding_length/num_heads】,
        （2）把seq_len, num_heads这两个维度转置
        【batch_size, num_heads，seq_len，embeding_length/num_heads】
        """
        x = tf.reshape(x, [batch_size,    -1,    self.num_heads,     self.depth])
        return tf.transpose(x,  [0,2,1,3])
    
    def call(self, q, k, v, training):
        batch_size = tf.shape(q)[0]
        
        q = self.q_linear_mapping(q, training=training)  #  (batch_size, seq_len, d_model)
        k = self.k_linear_mapping(k, training=training)  #  (batch_size, seq_len, d_model)
        v = self.v_linear_mapping(v, training=training)  #  (batch_size, seq_len, d_model)
        
        q = self.split_heads(q, batch_size)  #  (batch_size, num_heads, seq_len, depth)
        k = self.split_heads(k, batch_size)  #  (batch_size, num_heads, seq_len, depth)
        v = self.split_heads(v, batch_size)  #  (batch_size, num_heads, seq_len, depth)
        
        attention_out, attention_weights = scaled_dot_product_attention(q, k, v)
        # attention_out     =  [batch, num_heads, seq_len, depth]
        # attention_weights = [batch, num_heads, seq_len, seq_len]
        
        attention_out = tf.transpose(attention_out, [0,2,1,3])
        #  (batch_size, seq_len, num_heads,  depth)
        
        attention_out = tf.reshape(attention_out, [batch_size, -1,  self.d_model])
        #  (batch_size, seq_len, d_model)
        
        output = self.out_linear_mapping(attention_out, training=training)  #  (batch_size, seq_len, d_model)
        return  output, attention_weights


if __name__ == '__main__':
    temp_mha = MultiHeadAttention(d_model=16, num_heads=2)
    y = tf.random.uniform((1, 100, 16))  # (batch_size, encoder_sequence, d_model)
    att_out, aw = temp_mha(q=y, k=y, v=y)
    print('att_out.shape, aw.shape', att_out.shape, aw.shape )
    


#%% 前馈-非线性映射
class FeedForwardLayer(tf.keras.layers.Layer):
    def __init__(self, d_model,  dff ):
        super(FeedForwardLayer, self).__init__()
        
        self.ff1 = tf.keras.layers.Dense(dff, activation='gelu')  # (batch_size, seq_len, dff)
        self.ff2 = tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
        
    
    def call(self, x, training ):
        x1 = self.ff1(x,  training=training)
        x2 = self.ff2(x1, training=training)

        return x2
    
if __name__ == '__main__':
    sample_ffn = FeedForwardLayer(d_model=16, dff=64)
    sample_input = tf.random.uniform((1, 100, 16))
    sample_ffn_out = sample_ffn( sample_input, training=False)
    print('sample_ffn_out.shape ', sample_ffn_out.shape )

#%% Encoder层
# 22-9-6
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1 ):
        super(EncoderLayer, self).__init__()
        
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForwardLayer(d_model, 2*d_model)
        
        self.layernorm1 = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-6,)
        self.layernorm2 = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-6,)
        
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
    
    def call(self, x, training ):      
        
        # assert inputs.ndim == 3, f'Expected (batch, seq, hidden) got {inputs.shape}'
        # x = nn.LayerNorm(dtype=self.dtype)(inputs)
        # x = nn.MultiHeadDotProductAttention(
        #     dtype=self.dtype,
        #     kernel_init=nn.initializers.xavier_uniform(),
        #     broadcast_dropout=False,
        #     deterministic=deterministic,
        #     dropout_rate=self.attention_dropout_rate,
        #     num_heads=self.num_heads)(
        #         x, x)
        # x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        # x = x + inputs
    
        # # MLP block.
        # y = nn.LayerNorm(dtype=self.dtype)(x)
        # y = MlpBlock(
        #     mlp_dim=self.mlp_dim, dtype=self.dtype, dropout_rate=self.dropout_rate)(
        #         y, deterministic=deterministic)
    
        # return x + y

        # ViT: Norm > MHA > Add > Norm > MLP > Add
        norm_x          = self.layernorm1( x  )        # [batch, seq_len, d_model]
        #  之所以写成 norm_x，是为了在 计算 out1 时，需要使用原始输入，而不是norm后的数据
        att_out, aw = self.mha(q=norm_x, k=norm_x, v=norm_x,  training=training )
        att_out     = self.dropout1(att_out, training=training)
        # attention_out     =  [batch, seq_len, d_model]
        # attention_weights = [batch, num_heads, seq_len, seq_len]
        # print('att_out.shape, x.shape',att_out.shape, x.shape)
        out1       = att_out + x
        
        norm_out1  = self.layernorm2( out1, training=training   ) 
        ffn_out    = self.ffn( norm_out1,   training=training    )
        ffn_out    = self.dropout2(ffn_out, training=training)
        out2       = ffn_out + out1
        
        return out2, aw
    
if __name__ == '__main__':    
    sample_encoder_layer = EncoderLayer(d_model=16, num_heads=2, dff=64)
    
    sample_encoder_layer_output, sample_encoder_layer_aw = sample_encoder_layer(
        tf.random.uniform((1, 100, 16)), False )
    
    print('sample_encoder_layer_output.shape ', sample_encoder_layer_output.shape ) # (batch_size, input_seq_len, d_model)
    print('sample_encoder_layer_aw.shape ', sample_encoder_layer_aw.shape )


#%% Linear_1D
def get_Linear_Embedding1D(input_shape = (2048,1),  patch_size=(32,), d_model=64,):
    inputs1 = tf.keras.Input(shape =  (input_shape), name='inputs1')
    [length, depth] = input_shape
      
    out = tf.keras.layers.Conv1D(filters=d_model,  
                                kernel_size=patch_size, 
                                strides=patch_size, 
                                padding='same',
                                activation=None,             # 线性映射，不用激活函数
                                name='Linear_embed1D_c1')(inputs1)

    model = tf.keras.Model(inputs=inputs1, outputs=out, name='Linear_2D')
    return model

if __name__ == '__main__':
    LE1d = get_Linear_Embedding1D( )
    # LE1d.summary()
    print('Linear Embed1D', LE1d.input_shape, LE1d.output_shape, LE1d.count_params())

#%% Linear_2D
def get_Linear_Embedding2D(input_shape = (256,64),  patch_size=(8,8), d_model=64,):
    inputs1 = tf.keras.Input(shape =  (input_shape), name='inputs1')
    [length, depth] = input_shape
    
    xr = tf.keras.layers.Reshape(  (length, depth, 1),name='reshape_to_3d')(inputs1)
    
    x1 = tf.keras.layers.Conv2D(filters=d_model,  
                                kernel_size=patch_size, 
                                strides=patch_size, 
                                padding='same',
                                activation=None,          # 线性映射，不用激活函数
                                name='Linear_embed2D_c1')(xr)
    

    out = tf.keras.layers.Reshape((-1,d_model),name='Reshape_to_POSxDEP')(x1)

    model = tf.keras.Model(inputs=inputs1, outputs=out, name='Linear_2D')
    return model

if __name__ == '__main__':
    LE2d = get_Linear_Embedding2D(input_shape = (256,64),  patch_size=(8,8), d_model=64,)
    # LE1d.summary()
    print('Linear Embed2D', LE2d.input_shape, LE2d.output_shape, LE2d.count_params())
#%% 微型-表征网络
def get_TinyRN(input_shape = (256,64),  d_model=64,):
    inputs1 = tf.keras.Input(shape =  (input_shape), name='inputs1')
    [length, depth] = input_shape
    
    xr = tf.keras.layers.Reshape(  (length, depth, 1),name='reshape_to_3d')(inputs1)
    
    
    
    x1 = tf.keras.layers.Conv2D(filters=d_model,  kernel_size=(5,5), strides=(2, 2), 
                                padding='same',activation='gelu',name='c1')(xr)
    x1 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, name='x1p')(x1)


    x2 = tf.keras.layers.Conv2D(filters=d_model, kernel_size=(3,3), strides=(1, 1), 
                                padding='same',activation='gelu',name='c2')(x1)
    x2 = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-6,)(x2)
    x2 = tf.keras.layers.MaxPool2D(name='x2p')(x2)


    out = tf.keras.layers.Reshape((-1,d_model),name='Reshape_to_POSxDEP')(x2)

    model = tf.keras.Model(inputs=inputs1, outputs=out, name='TRN')
    return model

def get_TinyRN_1D(input_shape = (2048,1),  d_model=64,):
    inputs1 = tf.keras.Input(shape =  (input_shape), name='inputs1')
    [length, depth] = input_shape
    x1 = tf.keras.layers.Conv1D(filters=d_model,  kernel_size=11, strides=4, 
                                padding='same',activation='gelu',name='c1')(inputs1)
    x1 = tf.keras.layers.MaxPool1D( name='x1p')(x1)


    x2 = tf.keras.layers.Conv1D(filters=d_model, kernel_size=5, strides=2, 
                                padding='same',activation='gelu',name='c2')(x1)
    x2 = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-6,)(x2)
    x2 = tf.keras.layers.MaxPool1D(name='x2p')(x2)

    model = tf.keras.Model(inputs=inputs1, outputs=x2, name='TRN')
    return model

if __name__ == '__main__':
    TRN = get_TinyRN(input_shape = (256,64), d_model=64,)
    # TRN.summary()
    print('Tiny Represen', TRN.input_shape, TRN.output_shape, TRN.count_params())
    
    TRN1D = get_TinyRN_1D(input_shape = (2048,1), d_model=64,)
    # TRN.summary()
    print('TRN1D Represen', TRN1D.input_shape, TRN1D.output_shape, TRN1D.count_params())

#%%  
def get_AE(input_shape = (2048,1),  d_model=64,):
    inputs1 = tf.keras.Input(shape =  (input_shape), name='inputs1')
    [length, depth] = input_shape
    
    ################# Encoder ##########################
    d1 = tf.keras.layers.Conv1D(
                filters=d_model,  kernel_size=25, strides=4, 
                padding='same',activation='gelu',name='d1')(inputs1)
    # d1 = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-6,name='dN1')(d1)
    d1 = tf.keras.layers.MaxPool1D( name='d1p')(d1)
    
    
    d2 = tf.keras.layers.Conv1D(
                filters=d_model,  kernel_size=11, strides=2, 
                padding='same',activation='gelu',name='d2')(d1)
    d2 = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-6,name='dN2')(d2)
    d2 = tf.keras.layers.MaxPool1D(name='x2p')(d2)

#     print(d3.shape)
    ################# Encoder ##########################

    
    ################# Decoder ##########################
    # u2 = tf.keras.layers.Conv1DTranspose(
    #             filters=d_model, kernel_size=11, strides=4, 
    #             padding='same',activation='gelu',name='u2')(d2)
    # u2 = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-6,name='uN2')(u2)
    
#     a1 = tf.keras.layers.Concatenate(axis=-1,name='a1')([d1,u2])
    u1 = tf.keras.layers.Conv1DTranspose(
                filters=1, kernel_size=96, strides=32, 
                padding='same',activation=None,name='u1')(d2)
    # u1 = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-6,name='uN1')(u1)
    ################# Decoder ##########################

    model = tf.keras.Model(inputs=inputs1, outputs=[d2,u1], name='AE')
    return model

if __name__ == '__main__':
    AE = get_AE(input_shape = (2048,1), d_model=64,)
    AE.summary()
    print('AE Represen', AE.input_shape, AE.output_shape, AE.count_params())
    tf.keras.utils.plot_model(AE,to_file='AE.png',  show_shapes=0,)
#%% TransformerEncoder
# 22-9-6

class AutoEmbeddingTransformer(tf.keras.Model):
    '''Transformer with a Tiny Representation Network'''
    '''depth上的注意力用来查找敏感的通道，（这里指的是 CWT_Matrix=【时间，频率】 上的某个频率）'''
    def __init__(self, 
                 # 输入 2D信号 的参数
                 cwt_data   = False,
                 cwt_shape  = (256,64), 
                 cwt_patch  = (8,8),
                 cwt_embedding_way = 'Linear',  
                 # 'Linear'      = 以固定的patch去滑窗二维卷积映射
                 # 'NonLinear'   = 2层小核卷积（和池化）非线性映射
                 
                 # 输入 1D信号 的参数
                 sig_data   = False,
                 sig_shape  = (2048,1),
                 sig_patch  = (64,),
                 sig_embedding_way = 'Linear',
                 # 'Linear'      = 以固定的patch去滑窗二维卷积映射
                 # 'NonLinear'   = 2层小核卷积（和池化）非线性映射
                 # ‘AutoEncoder’ = 利用自编码器的编码做映射

                 # Transformer的参数
                 positional_encoding_way = 'trainable',
                 num_layers  = 6,   
                 cls_token   = True,
                 
                 # Encoder_Layer 的参数
                 d_model     = 64, 
                 num_heads   = 4, 
                 rate        = 0.1, 
                 
                 need_show   = False,
                 ):
        
        
        super(AutoEmbeddingTransformer, self).__init__()
        if  cwt_data==False and sig_data==False:
            print(cwt_data, sig_data,'No data input')
            assert 0
        
        self.cwt_data  = cwt_data
        self.cwt_shape = cwt_shape
        self.cwt_patch = cwt_patch
        self.cwt_embedding_way = cwt_embedding_way
        
        # 输入 1D信号 的参数
        self.sig_data   = sig_data
        self.sig_shape  = sig_shape
        self.sig_patch  = sig_patch
        self.sig_embedding_way = sig_embedding_way

        # Transformer的参数
        self.positional_encoding_way = positional_encoding_way
        self.num_layers  = num_layers
        self.cls_token   = cls_token
        
        # Encoder_Layer 的参数
        self.d_model     = d_model
        self.num_heads   = num_heads
        self.rate        = rate

        
        
        '''22-10-10-CWT 映射方式， 不用管存不存在CWT，直接定义，call里面决定用不用就完事了,
        上述做法不可以，定义了2中embed，一种用了，一种没有用，或者2个都没有用，
        在训练时，没被使用的embed模型就会报错说，没有梯度
        '''
        if cwt_data: 
            if cwt_embedding_way=='Linear':
                assert len(cwt_patch) == 2
                assert cwt_shape[0] % cwt_patch[0] == 0
                assert cwt_shape[1] % cwt_patch[1] == 0
                self.cwt_embedding    = get_Linear_Embedding2D(
                                        input_shape = cwt_shape,  
                                        patch_size  = cwt_patch, 
                                        d_model     = d_model,) 
                self.cwt_position_num = self.cwt_embedding.output_shape[1]
            elif cwt_embedding_way=='NonLinear':
                self.cwt_embedding    = get_TinyRN( 
                                        input_shape = cwt_shape, 
                                        d_model     = d_model,)
                self.cwt_position_num = self.cwt_embedding.output_shape[1]
        else:
            self.cwt_position_num = 0
        
        '''Signal 映射方式， 不用管存不存在 Signal ，直接定义，call里面决定用不用就完事了'''
        assert sig_embedding_way in ['Linear', 'NonLinear', 'AutoEncoder']
        if sig_data:
            if sig_embedding_way == 'Linear':
                assert len(sig_patch) == 1
                assert sig_shape[0] % sig_patch[0] == 0
                self.sig_embedding    = get_Linear_Embedding1D(
                                        input_shape = sig_shape,  
                                        patch_size  = sig_patch, 
                                        d_model     = d_model,) 
                self.sig_position_num = self.sig_embedding.output_shape[1]
            elif sig_embedding_way == 'NonLinear':
                self.sig_embedding    = get_TinyRN_1D( 
                                        input_shape = sig_shape, 
                                        d_model     = d_model,)
                self.sig_position_num = self.sig_embedding.output_shape[1]
            elif sig_embedding_way == 'AutoEncoder':
                self.sig_embedding    = get_AE( 
                                        input_shape = sig_shape, 
                                        d_model     = d_model,)
                # [(None, 64, 64), (None, 2048, 1)]
                self.sig_position_num = self.sig_embedding.output_shape[0][1]
        else:
            self.sig_position_num = 0
        
        self.num_positions = 0
        if cwt_data:
            self.num_positions += self.cwt_position_num
        if sig_data:
            self.num_positions += self.sig_position_num
        
        
        '''class token'''
        if cls_token: 
            self.num_positions += 1
            self.get_cls_token = tf.keras.layers.Dense(
                                            self.d_model,
                                            activation='sigmoid',
                                            use_bias=False,
                                            name='get_cls_token')
            self.out_shape  = (None, self.d_model)
        else:
            self.out_shape  = (None, self.num_positions)
        
        
        
        '''位置编码'''
        if self.positional_encoding_way not in [None, 'sin', 'trainable']:
            print(self.positional_encoding_way)
            assert 0
        if self.positional_encoding_way == 'sin':
            self.pos_encoding1  = positional_encoding(self.num_positions, self.d_model)       

        if self.positional_encoding_way == 'trainable':
            self.trainable_pos_encoding  = tf.keras.layers.Dense(   
                                            self.num_positions* self.d_model,
                                            activation='sigmoid', 
                                            use_bias=False,
                                            name='pos_encoding1') # [p*d]
            
        
        '''输入transformer encoder之前，把添加了位置编码的数据 Dropout'''
        self.drop_before_encod = tf.keras.layers.Dropout(  rate, name='Drop_before_encod_1' )
        
        
        '''位置注意力， 数据【pos, depth】 '''
        self.enc_layers    = [EncoderLayer( d_model   = self.d_model,
                                             num_heads = self.num_heads, 
                                             dff       = self.d_model * 4, 
                                             rate      = rate,
                                             # name      = 'pos_encoding_layer_{:02d}'.format(i), 
                                             ) 
                                                  for i in  range(  self.num_layers   ) ]
        
        '''特征 标准化'''
        self.feature_norm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-6,name='feature_norm')
        
        if need_show:
            print('''
                  # 输入 2D信号 的参数
                  cwt_data                \t = \t {},
                  cwt_shape               \t = \t {},
                  cwt_patch               \t = \t {},
                  cwt_embedding_way       \t = \t {},
                  
                  # 输入 1D信号 的参数
                  sig_data                \t = \t {},
                  sig_shape               \t = \t {},
                  sig_patch               \t = \t {},
                  sig_embedding_way       \t = \t {},
    
                  # Transformer的参数
                  positional_encoding_way \t = \t {},
                  num_layers              \t = \t {},
                  cls_token               \t = \t {},
                  
                  # Encoder_Layer 的参数
                  d_model                 \t = \t {},
                  num_heads               \t = \t {},
                  rate                    \t = \t {},  
                  
                  cwt_position_num        \t = \t {},
                  sig_position_num        \t = \t {},
                  num_positions           \t = \t {},
                  out_shape               \t = \t {}
                  '''.format(
                  # 输入 2D信号 的参数
                  cwt_data,
                  cwt_shape, 
                  cwt_patch ,
                  cwt_embedding_way,  
                  
                  # 输入 1D信号 的参数
                  sig_data,
                  sig_shape,
                  sig_patch,
                  sig_embedding_way,  
    
                  # Transformer的参数
                  positional_encoding_way,
                  num_layers,   
                  cls_token,
                  
                  # Encoder_Layer 的参数
                  d_model, 
                  num_heads, 
                  rate,
                  
                  self.cwt_position_num,
                  self.sig_position_num,
                  self.num_positions,    
                  self.out_shape,    
                  ))
    
    
    def call(self, x, training, need_w=False):
        bs, time_length, depth = tf.shape(x)
        # print(111,bs, time_length, depth)
        
        '''INPUT EMBEDDING'''
        if self.cwt_data and self.sig_data:
            assert time_length == 256+2048//64   # 256(jpg) + 32(sig2048//64)
            c = x[:,:256]
            s = x[:,256:]
            s =  tf.reshape(s, (bs,2048,1))
            # print('call',x.shape, c.shape, s.shape)
            
            ce = self.cwt_embedding(c, training=training)     # [256,64]  >x8x8 > [32, 8, 64]  >  [256,  64]  
            se = self.sig_embedding(s, training=training)     # [2048,1]  >x32 > [64,64] 
            # print('call',x.shape, c.shape, s.shape)
            xe = tf.concat([ce,se], axis=1)                   # [256,  64] + [64,64] = [256+64,64] 
            
        elif self.cwt_data:
            xe = self.cwt_embedding(x, training=training)     # [256,64]  >x8x8 > [32, 8, 64]  >  [256,  64]  
        elif self.sig_data:
            if self.sig_embedding_way=='AutoEncoder':
                xe, xrc = self.sig_embedding(x, training=training)    
                # xe  = embeded x = [2048,1]  >x32 > [64,64] 
                # xrc = reconstructed x = [2048,1]  >x32 > [64,64] >x32 >[2048,1] 
            else:
                xe = self.sig_embedding(x, training=training)    # [2048,1]  >x32 > [64,64] 
        else:
            print('No data input')
            assert 0

        
        '''CLASS TOKEN'''
        batch_size = tf.shape(x)[0]       
        ones = tf.ones(shape=(batch_size, 1, 1),  dtype=tf.float32)    # [bs, 1]
        if self.cls_token:
            cls_token = self.get_cls_token(ones)                       # [bs, 1, dm]
            # print('cls_token.shape', cls_token.shape)
            xe = tf.concat(values=(cls_token, xe), axis=1)             # [bs, 1, dm] concat [bs, pos, dm]
            # print('Add cls_token', xe.shape)
                
        '''位置编码'''
        if self.positional_encoding_way is None:
            pe = 0.0
            
        if self.positional_encoding_way == 'sin':
            pe = self.pos_encoding

        if self.positional_encoding_way == 'trainable':
            pe = self.trainable_pos_encoding(ones) # [bs, p*d]
            pe = tf.reshape(tensor=pe, shape=(-1, self.num_positions, self.d_model      )) # [bs, pos, d_model]
            
        
        '''位置注意力,时间维度'''
        # print(222,xe.shape,pe.shape)
        attention_weights = {}
        # print(123, x.shape, pe.shape)
        x = xe*0.8 + pe*0.2  # x的权重为0.8，positional encoding的权重为0.2，弱化编码的影响，强化对数据特征的学习
        x = self.drop_before_encod(x, training=training)
        for i in range(self.num_layers):
            # print('x1.shape',x1.shape)
            x, aw = self.enc_layers[i](x, training )
            attention_weights['pos_layer_{:d}'.format(i)] = aw  # positional attention weights
        # print('x.shape', x.shape)
        # print('attention_weights.keys()', attention_weights.keys())
        # print('attention_weights[pos_layer_0].shape', attention_weights['pos_layer_0'].shape)
        
        
        # [bs, pos, dep]
        # print(123,x.shape)
        feature = x[:,0,:] if self.cls_token else tf.reduce_mean(x, 1)
        
        # print(321, feature.shape)
        
        feature = self.feature_norm( feature )
        if self.sig_embedding_way=='AutoEncoder':
            feature = (xrc, feature)
        
        if need_w:
            return feature, attention_weights
        else:
            return feature
 

#%%

def get_ST(num_layers,need_show=False):
    '''# 这个是死的,里面的所有参数与《2022-重庆大学-TIM-可解释-S
    ignal-Transformer_A_Robust_and_Interpretable_Method_for_
    Rotating_Machinery_Intelligent_Fault_Diagnosis_Under_Variable_Operating_Conditions》
    保持一致
    '''

    model = AutoEmbeddingTransformer(            
                            # 输入 1D信号 的参数
                            sig_data   = True,
                            sig_shape  = (2048,1),
                            sig_patch  = (32,),
                            sig_embedding_way = 'Linear',   
                            
                            num_layers  = num_layers,
                            
                            need_show = need_show,
                            )
    model._name = 'ST'
    return model

if __name__ == '__main__':
    temp_model = get_ST(num_layers=1,need_show=True)
    
    temp_model_output, attention_weights = temp_model(
                                        x        = np.ones((7,2048,1)), 
                                        training = True, 
                                        need_w   = True )
    
    for key in attention_weights.keys():
        print(key,  attention_weights[key].shape )
    
    print( temp_model.name, len(temp_model_output) ,temp_model_output[0].shape,  temp_model_output[1].shape,  temp_model.count_params() )
    temp_model.summary()
#%%
def get_NET(num_layers,need_show=False):
    '''用卷积和池化实现非线性Embedding'''
    model = AutoEmbeddingTransformer(            # 输入 1D信号 的参数
                            sig_data   = True,
                            sig_shape  = (2048,1),
                            sig_patch  = (32,),
                            sig_embedding_way = 'NonLinear',   
                            
                            num_layers = num_layers,
                            need_show = need_show,
                            )
    model._name = 'NET'
    return model

if __name__ == '__main__':    
    temp_model = get_NET(num_layers=1, need_show=True)
    temp_model_output, attention_weights = temp_model(
                                        x        = np.ones((7,2048,1)), 
                                        training = True, 
                                        need_w   = True )
    
    for key in attention_weights.keys():
        print(key,  attention_weights[key].shape )
    
    print( temp_model.name, len(temp_model_output) ,temp_model_output[0].shape,  temp_model_output[1].shape,  temp_model.count_params() )
    
#%%
def get_AET(num_layers, need_show=False):
    '''自监督方式训练Embedding，'''
    model = AutoEmbeddingTransformer(
                            # 输入 1D信号 的参数
                            sig_data   = True,
                            sig_shape  = (2048,1),
                            sig_patch  = (32,),
                            sig_embedding_way = 'AutoEncoder',  
                            
                            num_layers = num_layers,
                            need_show = need_show,
                            )
    model._name = 'AET'
    return model

if __name__ == '__main__':     
    temp_model = get_AET(num_layers=1, need_show=True)
    temp_model_output, attention_weights = temp_model(
                                        x        = np.ones((7,2048,1)), 
                                        training = True, 
                                        need_w   = True )
    
    for key in attention_weights.keys():
        print(key,  attention_weights[key].shape )
    
    print( temp_model.name, len(temp_model_output) ,temp_model_output[0].shape,  temp_model_output[1].shape,  temp_model.count_params() )
    temp_model.summary()

