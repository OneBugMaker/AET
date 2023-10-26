# 22-9-9
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt



def positional_encoding(position = 50, d_model = 512):
    pos = np.arange(position)         # [50, ]
    idx = np.arange(d_model)          # [512,]
#     print(pos.shape,pos[:5],pos[-5:])
#     print(idx.shape,idx[:5],idx[-5:])
    
    angle_rates = 1 / np.power(10000, (2 * (idx//2)) / np.float32(d_model))  # [512]
#     print(angle_rates.shape)
    angle_rads = pos.reshape(-1,1) * angle_rates.reshape(1,-1)   # [50, 1]*[1, 512] >>> [50,512]
#     print(angle_rads.shape)
          
    # 将 sin 应用于数组中的偶数索引（indices）；2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # 将 cos 应用于数组中的奇数索引；2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)




#%%

if __name__ == '__main__':
   pos_encoding = positional_encoding(128, 16)
   print (pos_encoding.shape)

   plt.pcolormesh(pos_encoding[0], cmap='RdBu')
   plt.xlabel('Depth')
   # plt.xlim((0, 512))
   plt.ylabel('Position')
   plt.colorbar()
   plt.show()