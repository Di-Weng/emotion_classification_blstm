# -*- coding: utf-8 -*-
"""
-------------------------------
   Time    : 2018-12-02 12:29
   Author  : diw
   Email   : di.W@hotmail.com
   File    : predict.py
   Desc:   Load model, predict audio's class.
-------------------------------
"""

"""
audio_class = ['angry','fear','happy','neutral','sad','surprise']

Input audio's format: 
BIT DEPTH = 16（paInt16）
Sample Rate = 16000
CHANNELS = 1
 
Usage：
Load model: 
model = load_model('model/best_model.h5')

get_result: 
predict_class,predict_prob = get_audioclass(model,wav_file_path):

get_allaudio:  
predict_class,predict_prob,result_dic = get_audioclass(model,wav_file_path,all = True):
result_dic: {class:prob}

"""

from keras.models import load_model
from pyAudioAnalysis import audioFeatureExtraction
from keras.preprocessing import sequence
import numpy as np
from scipy import stats
import pickle
import librosa
import os

#获取音频
from get_audio import microphone_audio

classes = {0: 'angry', 1: 'fear', 2: 'happy', 3: 'neutral', 4: 'sad', 5: 'surprise'}
classes_e_n = {0: 'emotional', 1: 'neutral'}
# classes = {0: 'angry', 1: 'fear', 2: 'happy', 3: 'sad', 4: 'surprise'}
gender_classes = {0:'male',1:'female'}

max_len = 1024
nb_features = 36
nb_attention_param = 256
attention_init_value = 1.0 / 256
nb_hidden_units = 512   # number of hidden layer units
dropout_rate = 0.5
nb_lstm_cells = 128
nb_classes = 6

masking_value = -100.0

frame_size = 0.025  # 25 msec segments
step = 0.01     # 10 msec time step


def get_data(audio_path):
    # 采样率16000
    data, sr = librosa.load(audio_path, sr=16000)
    return data, sr


def extract_dataset_tosequence(data, Fs=16000, save=False):
    # x:librosa读取的文件 Fs:采样率
    f_global = []
    # 34D short-term feature
    f = audioFeatureExtraction.stFeatureExtraction(
        data, Fs, frame_size * Fs, step * Fs)

    # for pyAudioAnalysis which support python3
    if type(f) is tuple:
        f = f[0]

    # Harmonic ratio and pitch, 2D
    hr_pitch = audioFeatureExtraction.stFeatureSpeed(
        data, Fs, frame_size * Fs, step * Fs)
    f = np.append(f, hr_pitch.transpose(), axis=0)

    # Z-normalized
    f = stats.zscore(f, axis=0)

    f = f.transpose()
    f_global.append(f)
    # print("Extracting features from data")

    f_global = sequence.pad_sequences(f_global,
                                      maxlen=max_len,
                                      dtype='float32',
                                      padding='post',
                                      value=masking_value)

    if save:
        print("Saving features to file...")
        pickle.dump(f, open('features.p', 'wb'))

    return f_global


def find_max(list_iter):
    # 返回：概率，类别id
    prob_list = list_iter[0]
    max = prob_list[0]
    index = 0
    for i in range(len(prob_list)):
        current_prob = prob_list[i]
        if(current_prob >= max):
            max = current_prob
            index = i
    return max, index

# test_folder = '/Users/diweng/github_project/keras_audio_classifier/data/test'
def test_model(model_path, test_folder, model_type = 'emotion'):

    model = load_model(model_path)
    emotion_list = os.listdir(test_folder)
    total = 0
    count = 0
    if (model_type == 'emotion'):
        for current_emotion in emotion_list:
            if (current_emotion == '.DS_Store' or current_emotion == '_desktop.ini'):
                continue

            emotion_total = 0
            emotion_count = 0
            current_emotion_path = test_folder + '/' + current_emotion
            test_file_list = os.listdir(current_emotion_path)
            for current_test_file in test_file_list:
                if (current_test_file == '.DS_Store' or current_test_file == '_desktop.ini'):
                    continue

                test_file_path = current_emotion_path + '/' + current_test_file
                data, sr = get_data(test_file_path)
                f = extract_dataset_tosequence(data, sr)
                f_ex = np.full((f.shape[0], nb_attention_param),
                               attention_init_value, dtype=np.float32)
                predict_output = model.predict([f_ex, f])
                predict_prob, predict_label = find_max(predict_output)
                predict_class = classes[predict_label]
                total += 1
                emotion_total += 1
                if(predict_class == current_emotion):
                    emotion_count += 1
                    count += 1
            current_accuracy = float(emotion_count) / emotion_total
            print('%s accuracy: %.2f%%' % (str(current_emotion), current_accuracy * 100))
    elif(model_type == 'gender'):
        speaker_class = {'ZhaoZuoxiang':0, 'wangzhe':0, 'zhaoquanyin':1, 'liuchanhg':1}
        for current_emotion in emotion_list:
            if (current_emotion == '.DS_Store' or current_emotion == '_desktop.ini'):
                continue

            gender_total = 0
            gender_count = 0
            current_emotion_path = test_folder + '/' + current_emotion
            test_file_list = os.listdir(current_emotion_path)
            for current_test_file in test_file_list:
                if (current_test_file == '.DS_Store' or current_test_file == '_desktop.ini'):
                    continue
                current_gender = speaker_class[current_test_file.split('_')[1]]
                test_file_path = current_emotion_path + '/' + current_test_file
                data, sr = get_data(test_file_path)
                f = extract_dataset_tosequence(data, sr)
                f_ex = np.full((f.shape[0], nb_attention_param),
                               attention_init_value, dtype=np.float32)
                predict_output = model.predict([f_ex, f])
                predict_prob, predict_label = find_max(predict_output)
                total += 1
                gender_total += 1
                if (predict_label == current_gender):
                    gender_count += 1
                    count += 1
            current_accuracy = float(gender_count) / gender_total
            print('%s accuracy: %.2f%%' % (str(current_gender), current_accuracy * 100))
    total_accuracy = float(count) / total
    print('Total accuracy: %.2f%%' % (total_accuracy * 100))

def analyse_emotionn(model,test_file):
    dic = {}
    data, sr = get_data(test_file)
    f = extract_dataset_tosequence(data, sr)
    f_ex = np.full((f.shape[0], nb_attention_param),
                   attention_init_value, dtype=np.float32)
    predict_output = model.predict([f_ex, f])
    predict_prob, predict_label = find_max(predict_output)
    predict_class = classes[predict_label]
    for i in range(len(predict_output[0])):
        current_prob = predict_output[0][i]
        current_class = classes[i]
        # print('当前语音的情感为：%-8s 的概率为：%.2f%%' %
              # (str(current_class), current_prob * 100))
        dic[current_class] = current_prob * 100
    # print('因此，当前语音的情感为：%s, 概率为：%.2f%%' %
          # (str(predict_class), predict_prob * 100))
    return dic

def get_audioclass(model,test_file,model_type = 'emotion',all = False):
    if(model_type == 'emotion'):
        data, sr = get_data(test_file)
        f = extract_dataset_tosequence(data, sr)
        f_ex = np.full((f.shape[0], nb_attention_param),
                       attention_init_value, dtype=np.float32)
        predict_output = model.predict([f_ex, f])
        predict_prob, predict_label = find_max(predict_output)
        predict_class = classes[predict_label]
        class_dic = {}
        for i in range(len(predict_output[0])):
            current_prob = predict_output[0][i]
            current_class = classes[i]
            class_dic[current_class] = current_prob
            # print('当前语音的情感为：%-8s 的概率为：%.2f%%' %
            #       (str(current_class), current_prob * 100))
        if(all):
            return predict_class,predict_prob,class_dic
        return predict_class,predict_prob
    elif(model_type == 'gender'):
        data, sr = get_data(test_file)
        f = extract_dataset_tosequence(data, sr)
        f_ex = np.full((f.shape[0], nb_attention_param),
                       attention_init_value, dtype=np.float32)
        predict_output = model.predict([f_ex, f])
        predict_prob, predict_label = find_max(predict_output)
        predict_class = gender_classes[predict_label]
        class_dic = {}
        for i in range(len(predict_output[0])):
            current_prob = predict_output[0][i]
            current_class = gender_classes[i]
            class_dic[current_class] = current_prob
            # print('当前语音的情感为：%-8s 的概率为：%.2f%%' %
            #       (str(current_class), current_prob * 100))
        if (all):
            return predict_class, predict_prob, class_dic
        return predict_class, predict_prob
    elif(model_type == 'emotion_neutral'):
        data, sr = get_data(test_file)
        f = extract_dataset_tosequence(data, sr)
        f_ex = np.full((f.shape[0], nb_attention_param),
                       attention_init_value, dtype=np.float32)
        predict_output = model.predict([f_ex, f])
        predict_prob, predict_label = find_max(predict_output)
        predict_class = classes_e_n[predict_label]
        return predict_prob,predict_class



if __name__ == '__main__':

    #input wav format
    # FORMAT = pyaudio.paInt16
    # CHANNELS = 1
    # RATE = 16000

    test_file = 'recordFiles/20181211164803_1595.wav'
    test_folder = '/Users/diweng/github_project/keras_audio_classifier/data/test'
    model_path = 'model/best_model.h5'
    model = load_model(model_path)

    # test_model(model_path,test_folder,model_type='gender')

    # #获取音频
    # microphone_audio(test_file)
    #
    #验证模型正确率
    # print(analyse_emotionn(model,test_file))
    emotion_predict_class, emotion_predict_prob, emotion_class_dic = get_audioclass(model, test_file, 'emotion',
                                                                                    all=True)
