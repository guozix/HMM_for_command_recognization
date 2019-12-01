# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 15:11:51 2019
"""

import pyaudio
import numpy as np
import struct
import wave
import os
from hmmlearn import hmm
from get_mfc_data import get_mfc_data

def open_file_mfc(filename):
    #输入文件名
    #返回
    '''
    读取mfcc文件，返回二维数组，表示每个时间点的mfcc向量
    '''
    mfcc_file = open(filename,"rb")
    data_bytes = mfcc_file.read()
    #print(data_bytes[:4])
    (frame_num,frame_cyc,mfcc_bytes,feakind) = struct.unpack(">iihh",data_bytes[:12])
    #帧数量，帧间隔，每帧mfcc数据bytes数
    #print(frame_num,frame_cyc,mfcc_bytes)
    
    mfcc = []
    frame_start = 12
    for i in range(frame_num):
        mfcc.append(list(struct.unpack(">"+'f'*39,data_bytes[frame_start:frame_start+mfcc_bytes])))
        frame_start += mfcc_bytes
    mfcc = np.array(mfcc)
    mfcc_file.close()
    print(mfcc.shape)
    #print(len(mfcc))
    return mfcc

def get_audio(filepath):
    ans = str(input("是否开始录音？(y/n)"))
    if ans == str("y") :
        CHUNK = 1000
        FORMAT = pyaudio.paInt16
        CHANNELS = 1                # 声道数
        RATE = 16000                # 采样率
        RECORD_SECONDS = 3
        WAVE_OUTPUT_FILENAME = filepath
        p = pyaudio.PyAudio()

        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        print("开始3秒录音")
        frames = []
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
        print("录音结束\n")

        stream.stop_stream()
        stream.close()
        p.terminate()

        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        return True
    else:
        return False

def getEnergy(wave_data):
    energy = []
    sum = 0
    for i in range(len(wave_data)) :
        sum = sum + (int(wave_data[i]) * int(wave_data[i]))
        if (i + 1) % 256 == 0 :
            energy.append(sum)
            sum = 0
        elif i == len(wave_data) - 1 :
            energy.append(sum)
    return energy

def sgn(a):
    if a>=0:
        return 1
    else :
        return -1
    
def pass_zero(wave_data):
    zero_rate = []
    sum = 0
    for i in range(len(wave_data)):
        if i == 0:
            sum = sum + np.abs(sgn(wave_data[i]))
        else :
            sum = sum + np.abs(sgn(wave_data[i])-sgn(wave_data[i-1]))
        if (i+1)%256 == 0:
            sum = sum/(2*256)
            zero_rate.append(sum)
            sum = 0
        elif i == (len(wave_data) -1):
            sum = sum/(2*256)
            zero_rate.append(sum)
    return zero_rate

def cut_sound(raw_data, file_en_path, file_zero_path):
    '''
    根据帧能量和过零率两个文件(后两个参数)判断语音段，并对第一个参数raw_data二进制数据进行切割
    '''
    #读取帧能量数值
    file_en = open(file_en_path,"r")
    all_data = file_en.read()
    file_en.close()
    sec_en = all_data.split('\n')
    sec_en = sec_en[:-1]
    sec_en = list(map(int,sec_en))
    
    #读取帧过零率数值
    file_en = open(file_zero_path,"r")
    all_data = file_en.read()
    file_en.close()
    sec_zero = all_data.split('\n')
    sec_zero = sec_zero[:-1]
    sec_zero = list(map(float,sec_zero))
        
    #生成判断阈值
    #取一定比例的数据作为计算标准
    n_shold = int(len(sec_en)*0.1)
    
    en_sort = np.argsort(sec_en)
    en_min = 0
    en_max = 0
    for idx_idx in range(n_shold):
        idx = en_sort[idx_idx]
        en_min += sec_en[idx]
    en_min = en_min/n_shold
    
    for idx_idx in range(len(sec_en)-n_shold,len(sec_en)):
        idx = en_sort[idx_idx]
        en_max += sec_en[idx]
    en_max = en_max/n_shold
    
    zero_sort = np.argsort(sec_zero)
    zero_min = 0
    zero_max = 0
    for idx_idx in range(n_shold):
        idx = zero_sort[idx_idx]
        zero_min += sec_zero[idx]
    zero_min = zero_min/n_shold
    
    for idx_idx in range(len(sec_zero)-n_shold,len(sec_zero)):
        idx = zero_sort[idx_idx]
        zero_max += sec_zero[idx]
    zero_max = zero_max/n_shold
    
    #print("min energy: {}".format(en_min))
    #print("max energy: {}".format(en_max))
    #print("min zero rate: {}".format(zero_min))
    #print("max zero rate: {}".format(zero_max))
    #选择性写入语音片段
    zero_min = 0.25
    #en_min = en_min+en_max*0.001
    en_min = 1000000
    #print("stage energy: {}".format(en_min))
    #print("stage zero rate: {}".format(zero_min))
    #raw_data_select = cut_speak(raw_data)
    speak_dom = []
    start_point = 0
    end_point = 0
    m = 8 # 生成判断的帧长度
    k=-1
    while k < (len(sec_en)-1):
        k+=1
        if (sec_en[k]>en_min)or(sec_zero[k]>zero_min):#检查到可能的起始点
            start_point=k
            if (np.all(np.array(sec_en[k:k+m])>en_min))or(np.all(np.array(sec_zero[k:k+m])>zero_min)):
                #确认起始点
                k+=m#跳过必然是语音的段
                while not ((np.all(np.array(sec_en[k:k+m])<en_min))and(np.all(np.array(sec_zero[k:k+m])<zero_min))):
                    if (k+m) < len(sec_en): 
                        k+=1
                    else:
                        end_point = k+m
                        break
                
                if not end_point:
                    end_point=k
                k+=m#跳过不可能是语音的段
                speak_dom.append((start_point,end_point))
                end_point = 0
  
    
    print(speak_dom)
    raw_data_select = b''
    for dom in speak_dom:
        raw_data_select += raw_data[dom[0]*256*2:dom[1]*256*2]
    
    file_en = wave.open("./temp_sound/input_voice.wav","wb")
    file_en.setnchannels(1)
    file_en.setsampwidth(2)
    file_en.setframerate(16000)
    file_en.writeframes(raw_data_select)
    file_en.close()


##main
if __name__ == '__main__':
    order_text = ['开门','关门','打开','关闭','安静']
    #训练
    #datas = get_mfc_data('C:/Users/18341/Desktop/book/听觉/实验3-语音识别/语料/features/')
    datas = get_mfc_data('F:/HIT/大三上/视听觉/lab3/组/gzx_sound_mfcc/')
    

    #model = hmm.GaussianHMM(n_components = 5, n_iter = 20, tol = 0.01, covariance_type="diag")

    hmms = dict()
    
    for category in datas:
        Qs = datas[category]
        n_hidden = 6
        model = hmm.GaussianHMM(n_components = 5, n_iter = 20, tol = 0.01, covariance_type="diag")
        vstack_Qs = np.vstack(tuple(Qs[:-3]))
        #print(tuple(Qs[:-3]))
        #print('----------')
        #print([Q.shape[0] for Q in Qs[:-3]])
        #print('-++++++++++++')
        model.fit(vstack_Qs, [Q.shape[0] for Q in Qs[:-3]])
        print('success fit')
        hmms[category] = model
        
    '''
    #test
    correct_num = 0
    for category in datas:
        for test_sample in datas[category][-3:]:
            print('real_category:', category)
            max_score = -1 * np.inf
            predict = -1
            for predict_category in hmms:
                model = hmms[predict_category]
                score = model.score(test_sample)
                print('category', predict_category, '. score:', score)
                if score > max_score:
                    max_score = score
                    predict = predict_category
                    #print('predict_category', predict_category)
            if predict == category:
                correct_num += 1
            print('predict_category:',predict)
    print(correct_num / (3*5))
    '''
    while(True):
        # 麦克风采集的语音输入
        input_filename = "input.wav"               
        input_filepath = './temp_sound/'
        in_path = input_filepath + input_filename
        rec_flag = get_audio(in_path) #声音采集并写入文件input.wav
        
        if not rec_flag:#如果没有语音输入，则不再识别
            break
        
        # 提取语音段
        f = wave.open(in_path,"rb")
        params = f.getparams()#返回所有的WAV文件的格式信息
        nchannels, sampwidth, framerate, nframes = params[:4]
        #nchannels声道, sampwidth量化比特数（字节数）, framerate采样频率, nframes总采样点数目
        
        #readframes()按照采样点读取数据
        raw_data = f.readframes(nframes)#bytes类型
        wave_data = np.frombuffer(raw_data, dtype = np.short)#int16类型的一维numpy array
        #转成二字节数组形式（每个采样点占两个字节）
        f.close()
        
        #计算能量和过零率
        energy = getEnergy(wave_data)
        file_en_path = "./temp_sound/input_en.txt"
        file_en = open(file_en_path,"w+") 
        for j in range(len(energy)):
            file_en.write(str(energy[j])+'\n')
        file_en.close()
        
        zero_rate = pass_zero(wave_data)
        file_zero_path = "./temp_sound/input_zero.txt"
        file_en = open(file_zero_path,"w+") 
        for j in range(len(zero_rate)):
            file_en.write(str(zero_rate[j])+'\n')
        file_en.close()
        
        cut_sound(raw_data, file_en_path, file_zero_path)
        
        order = "hcopy.exe -A -D -T 1 -C tr_wav.cfg -S list.scp"
        r_v = os.popen(order) #执行命令行
        
        test_sample = open_file_mfc('./temp_sound/input_mfcc.mfc')
        
        ##识别
        max_score = -1 * np.inf
        predict = -1
        for predict_category in hmms:
            model = hmms[predict_category]
            score = model.score(test_sample)
            print('category', predict_category, '. score:', score)
            if score > max_score:
                max_score = score
                predict = predict_category
                #print('predict_category', predict_category)
        print('predict_category:',predict)
        print(order_text[int(predict)-1])
    
    
    
    
