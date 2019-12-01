import struct
import numpy as np
import os
# input: 特征文件夹的路径
# return: datas, 各类别的音频特征组成的列表
# 其中datas['1'] = a list :属于类别1的特征列表, 列表中每个元素都是一个特征序列(T*n_dim的numpy矩阵)
def get_mfc_data(path):
    files = os.listdir(path)
    datas = dict()
    for file_name in files: # 读取每个mfc文件到矩阵data中
        data = list()
        with open(path+file_name, 'rb') as f:
            nframes = struct.unpack('>i',f.read(4))[0] # 帧数 
            _ = struct.unpack('>i',f.read(4))[0]   # 帧移，100ns为单位，100000指10ms，
            nbytes = struct.unpack('>h',f.read(2))[0]  # 每帧特征值的字节长度
            ndim = nbytes / 4                            # 每帧的特征的维度（一维为一个int）
            _ = struct.unpack('>h',f.read(2))[0] # [没用] 用户序号
            while True:
                data_byte = f.read(4)
                if len(data_byte) < 4: 
                    break
                data.append(struct.unpack('>f', data_byte)[0])   
        data = np.array(data)
        data.shape = nframes, int(ndim)
        category = file_name[0]
        if category in datas:
            datas[category].append(data)
        else:
            datas[category] = list()
            datas[category].append(data)
    return datas
    