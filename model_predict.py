# -*- coding: utf-8 -*-
# @Time    : 2018/7/10 10:46
# @Author  : MengnanChen
# @FileName: predict.py
# @Software: PyCharm Community Edition

import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'utility'))

from keras.models import load_model
import functions
import globalvars
import librosa
import numpy as np

emotion_classes = ['anger', 'boredom', 'disgust',
                   'anxiety(fear)', 'happiness', 'sadness', 'neutral']


def predict_class(data_path: str, current_model):
    y, sr = librosa.load(data_path, sr=16000)  # librosa:load wav
    f = functions.feature_extract_test(data=(y, sr))  # feature extraction
    u = np.full((f.shape[0], globalvars.nb_attention_param),
                globalvars.attention_init_value, dtype=np.float64)

    # the shape of result is [1,7], e.g.:[[0.31214175 0.04727687 0.01413498 0.13356456 0.4746141  0.00477368 0.01349405]]
    result = current_model.predict([u, f])
    # print('type of result:',type(result)) # <class 'numpy.ndarray'>
    result_dic = {}
    current_index = 0
    for current_prob in result[0]:
        result_dic[emotion_classes[current_index]] = current_prob
        current_index += 1
    return result_dic

if __name__ == '__main__':
    model_path = 'model/berlin.h5'
    model = load_model(model_path)
    data_path = '/Users/diweng/github_project/keras_audio_classifier/data/test/happy/相同文本300_liuchanhg_happy_201.wav'
    result_dic = predict_class(data_path, model)
    print(result_dic)

