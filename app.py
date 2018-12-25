# all the imports
import sqlite3
from flask import Flask, request, session, g, redirect, url_for, \
     abort, render_template, flash
from contextlib import closing
import os
import datetime
import tensorflow as tf
import random
from predict import get_audioclass
import numpy as np
from flask import Markup
import json
from keras.models import load_model
import pymongo
from keras import backend as K
from OpenSSL import SSL
from predict import classes
from model_predict import predict_class
# set GPU memory
if('tensorflow' == K.backend()):
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

#app.config.from_envvar('FLASKR_SETTINGS', silent=True)

# configuration

DATABASE = 'flaskr.db'
DEBUG = True
SECRET_KEY = 'developmentkey'
USERNAME = 'admin'
PASSWORD = 'default'
emotion_model_path = 'model/berlin.h5'
gender_model_path = 'model/gender_model.h5'
emotion_neutral_model_prepath = 'model/emotion_neutral_model_'
collection_name = 'user_emotion_1'
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# 加载第一个模型
g1 = tf.Graph() # 加载到Session 1的graph
g2 = tf.Graph() # 加载到Session 2的graph


sess1 = tf.Session(graph=g1) # Session1
sess2 = tf.Session(graph=g2) # Session2


with sess1.as_default():
    with g1.as_default():
        emotion_model = load_model(emotion_model_path)

# 加载第二个模型
with sess2.as_default():
    with g2.as_default():
        gender_model = load_model(gender_model_path)

app = Flask(__name__)
app.config.from_object(__name__)

def connect_db():
    rv = sqlite3.connect(app.config['DATABASE'])
    rv.row_factory = sqlite3.Row
    return rv

def init_db():
    with app.app_context():
        db = get_db()
        with app.open_resource('schema.sql', mode='r') as f:
            db.cursor().executescript(f.read())
        db.commit()

def get_db():
    if not hasattr(g,'sqlite_db'):
        g.sqlite_db = connect_db()
    return g.sqlite_db

def close_db(error):
    if hasattr(g,'sqlite_db'):
        g.sqlite_db.close()

def sox_noiseclean(file_inputpath,noise_filepath,outputfile_path):
    extract_noise = 'sox '+ APP_ROOT + '/' + file_inputpath + ' -t null /dev/null trim 0 1 noiseprof ' + APP_ROOT + '/' + noise_filepath
    clean_noise = 'sox ' + APP_ROOT + '/' + file_inputpath + ' ' + APP_ROOT + '/' +  outputfile_path + ' noisered ' + APP_ROOT + '/' + noise_filepath + ' 0.26'
    print(extract_noise)
    print(clean_noise)
    print(os.system(extract_noise))
    print(os.system(clean_noise))

# @app.route('/')
# def hello_world():
#     return 'Hello World!'


@app.before_request
def before_request():
    # init_db()
    g.db = connect_db()

@app.teardown_request
def teardown_request(exception):
    db = getattr(g, 'db', None)
    if db is not None:
        db.close()
    g.db.close()

@app.route('/')
@app.route('/index')
def show_index():
    return render_template('index.html')

@app.route('/show_demo')
def show_demo():
    return render_template('show_demo.html')
#127.0.0.1
#127.0.0.1
def conn_mongo(ServerURL = '192.168.12.120',db_name = 'user_sentiment'):
    conn = pymongo.MongoClient(ServerURL,
                               27017,
                               username='wd',
                               password='wd123456',
                               )
    db = conn[db_name]
    return db,conn



@app.route('/emo_visual')
def emo_visual():
    # cur = g.db.execute('select userName, use_date, angry, sad, fear, happy,surprise from user_sentiment')
    current_db,current_conn = conn_mongo()
    current_collection = current_db[collection_name]
    # lis = [dict(userName = row['userName'],use_date=row['use_date'],angry = row['angry'],sad = row['sad'],fear = row['fear'],happy = row['happy'],surprise = row['surprise']) for row in cur.fetchall()]
    lis = [dict(userName=current_data['userName'], use_date=current_data['use_date'],
                            angry=current_data['angry'], bored=current_data['bored'], disgust=current_data['disgust'],
                            anxious=current_data['anxious'],happy=current_data['happy'], sad = current_data['sad'],
                            neutral=current_data['neutral']) for current_data in
                       current_collection.find()]

    print(lis)
    current_conn.close()
    return render_template('emo_visual.html',lis = lis)



@app.route('/get_audio', methods=['GET', 'POST'])
def get_audio():
        return render_template('get_audio.html')

@app.route('/get_class/<string:saved>', methods=['GET', 'POST'])
def get_class(saved):
    if (saved == "0" or saved=="1"):
        if request.method == 'POST':
            userName=request.headers["userName"]
            timenow = datetime.datetime.now()
            date_time = datetime.datetime.strftime(timenow, '%Y%m%d%H%M%S')
            rand_int = random.randint(1, 10000)
            file_name = userName+"_"+date_time + "_" + str(rand_int)
            original_filename = "recordFiles/original/" + file_name + ".wav"
            # if(saved == "0"):
            request.files['audioData'].save(original_filename)
            print(file_name)

            noise_filename = "recordFiles/noise/" + file_name

            #使用降噪后的音频
            filename = "recordFiles/clean/" + file_name + ".wav"

            sox_noiseclean(original_filename,noise_filename,filename)


            # gender prediction
            with sess2.as_default():
                with sess2.graph.as_default():
                    gender_predict_class, gender_predict_prob, gender_class_dic = get_audioclass(gender_model,
                                                                                                 filename,
                                                                                                 'gender',
                                                                                                 all=True)
            jsonData = {}
            emotion_class_list = []
            emotion_prob_list = []
            gender_class_list = []
            gender_prob_list = []

            for current_gender_class, current_gender_prob in gender_class_dic.items():
                gender_class_list.append(current_gender_class)
                gender_prob_list.append(float(current_gender_prob))
            current_db, current_conn = conn_mongo()


            # ['anger', 'boredom', 'disgust','anxiety(fear)', 'happiness', 'sadness', 'neutral']
            with sess1.as_default():
                with sess1.graph.as_default():
                    emotion_class_dic = predict_class(filename, emotion_model)
                    # emotion_predict_class, emotion_predict_prob,emotion_class_dic  = get_audioclass(emotion_model, filename,'emotion', all=True)

            for current_emotion_class, current_emotion_prob in emotion_class_dic.items():
                emotion_class_list.append(current_emotion_class)
                emotion_prob_list.append(float(current_emotion_prob))

            if(saved=="0"):
                timeItem=str(datetime.datetime.strftime(timenow, '%Y-%m-%d %H:%M:%S'))
                data_mongo = {'userName':userName,'use_date':timeItem}
                # selfEmo = [userName, str(timeItem)]
                for item in emotion_class_dic:
                    # selfEmo.append(emotion_class_dic[item]*100)
                    data_mongo[item] = emotion_class_dic[item]*100
                    # print(selfEmo)
                    # print(data_mongo)


                current_collection = current_db[collection_name]
                result = current_collection.insert_one(data_mongo)
                # print(result)
                # ['angry', 'bored', 'disgust','anxious', 'happy', 'sad', 'neutral']
                lis = [dict(userName=current_data['userName'], use_date=current_data['use_date'],
                            angry=current_data['angry'], bored=current_data['bored'], disgust=current_data['disgust'],
                            anxious=current_data['anxious'],happy=current_data['happy'], sad = current_data['sad'],
                            neutral=current_data['neutral']) for current_data in
                       current_collection.find()]
                # print(lis)

            jsonData['emotion_class'] = emotion_class_list
            jsonData['emotion_prob'] = emotion_prob_list
            jsonData['gender_class'] = gender_class_list
            jsonData['gender_prob'] = gender_prob_list
            return_data = json.dumps(jsonData)
            current_conn.close()
            return (return_data)

if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=9593, ssl_context=('/Users/diweng/test/server.crt', '/Users/diweng/test/server.key'))
    context = ('/Users/diweng/test/server.crt','/Users/diweng/test/server.key')
    # app.run(host='0.0.0.0', port=9593)
    app.run(host='0.0.0.0', port=9593,ssl_context=context)
