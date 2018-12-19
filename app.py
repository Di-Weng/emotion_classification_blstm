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
emotion_model_path = 'model/best_model_2.h5'
gender_model_path = 'model/gender_model.h5'
emotion_neutral_model_prepath = 'model/emotion_neutral_model_'
collection_name = 'user_emotion_1'

# 加载第一个模型
g1 = tf.Graph() # 加载到Session 1的graph
g2 = tf.Graph() # 加载到Session 2的graph
g3 = tf.Graph() # 加载到Session 1的graph
g4 = tf.Graph() # 加载到Session 2的graph
g5 = tf.Graph() # 加载到Session 1的graph
g6 = tf.Graph() # 加载到Session 2的graph
g7 = tf.Graph() # 加载到Session 1的graph

sess1 = tf.Session(graph=g1) # Session1
sess2 = tf.Session(graph=g2) # Session2
sess3 = tf.Session(graph=g3) # Session1
sess4 = tf.Session(graph=g4) # Session2
sess5 = tf.Session(graph=g5) # Session1
sess6 = tf.Session(graph=g6) # Session2
sess7 = tf.Session(graph=g7) # Session1

with sess1.as_default():
    with g1.as_default():
        emotion_model = load_model(emotion_model_path)

# 加载第二个模型
with sess2.as_default():
    with g2.as_default():
        gender_model = load_model(gender_model_path)

with sess3.as_default():
    with g3.as_default():
        emotion_neutral_model_1 = load_model(emotion_neutral_model_prepath +'1.h5')

with sess4.as_default():
    with g4.as_default():
        emotion_neutral_model_2 = load_model(emotion_neutral_model_prepath +'2.h5')

with sess5.as_default():
    with g5.as_default():
        emotion_neutral_model_3 = load_model(emotion_neutral_model_prepath +'3.h5')

with sess6.as_default():
    with g6.as_default():
        emotion_neutral_model_4 = load_model(emotion_neutral_model_prepath +'4.h5')

with sess7.as_default():
    with g7.as_default():
        emotion_neutral_model_5 = load_model(emotion_neutral_model_prepath +'5.h5')

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
    lis = [dict(userName = current_data['userName'],use_date=current_data['use_date'],angry = current_data['angry'],sad = current_data['sad'],fear = current_data['fear'],happy = current_data['happy'],surprise = current_data['surprise']) for current_data in current_collection.find()]
    print(lis)
    current_conn.close()
    return render_template('emo_visual.html',lis = lis)



@app.route('/get_audio', methods=['GET', 'POST'])
def get_audio():
        return render_template('get_audio.html')

def whether_emotional(file_name):
    class_list = []
    prob_list = []
    emotion_count = 0
    # 声音数据为emotional的概率
    emotional_prob_list = []
    neutral_count = 0
    with sess3.as_default():
        with g3.as_default():
            class_prob,neutral_class = get_audioclass(emotion_neutral_model_1, file_name,'emotion_neutral')
            class_list.append(neutral_class)
            prob_list.append(class_prob)
    with sess4.as_default():
        with g4.as_default():
            class_prob,neutral_class = get_audioclass(emotion_neutral_model_2, file_name,'emotion_neutral')
            class_list.append(neutral_class)
            prob_list.append(class_prob)

    with sess5.as_default():
        with g5.as_default():
            class_prob,neutral_class = get_audioclass(emotion_neutral_model_3, file_name,'emotion_neutral')
            class_list.append(neutral_class)
            prob_list.append(class_prob)

    with sess6.as_default():
        with g6.as_default():
            class_prob,neutral_class = get_audioclass(emotion_neutral_model_4, file_name,'emotion_neutral')
            class_list.append(neutral_class)
            prob_list.append(class_prob)

    with sess7.as_default():
        with g7.as_default():
            class_prob,neutral_class = get_audioclass(emotion_neutral_model_5, file_name,'emotion_neutral')
            class_list.append(neutral_class)
            prob_list.append(class_prob)

    for i in range(len(class_list)):
        if(class_list[i]=='emotional'):
            emotion_count += 1
            emotional_prob_list.append(prob_list[i])
        else:
            neutral_count += 1
            emotional_prob_list.append(1-float(prob_list[i]))

    if(emotion_count > neutral_count):
        return ('emotional', emotional_prob_list)
    else:
        return ('neutral', emotional_prob_list)


@app.route('/get_class/<string:saved>', methods=['GET', 'POST'])
def get_class(saved):
    if (saved == "0" or saved=="1"):
        if request.method == 'POST':
            userName=request.headers["userName"]
            timenow = datetime.datetime.now()
            filename = "recordFiles/" + userName+"_"+datetime.datetime.strftime(timenow, '%Y%m%d%H%M%S') + "_" + str(
                random.randint(1, 10000)) + ".wav"
            # if(saved == "0"):
            request.files['audioData'].save(filename)
            print(filename)

            # list长度为5， 每个元素代表每个模型判断语音数据为emotional的概率
            # whether_emotional_str， emotional 或 neutral；前者为带情感， 后者为不带情感
            whether_emotional_str,emotional_probility_list = whether_emotional(filename)
            print(whether_emotional_str)
            print(emotional_probility_list)
            if(whether_emotional_str == 'emotional'):
                emoFlag = 1
            else:
                emoFlag = 0

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

            # emotion prediction
            if(emoFlag==1):
                with sess1.as_default():
                    with sess1.graph.as_default():
                        emotion_predict_class, emotion_predict_prob,emotion_class_dic  = get_audioclass(emotion_model, filename,
                                                                                                        'emotion', all=True)
                for current_emotion_class, current_emotion_prob in emotion_class_dic.items():
                    emotion_class_list.append(current_emotion_class)
                    emotion_prob_list.append(float(current_emotion_prob))
            else:
                emotion_class_list=['angry','fear','happy','neutral','sad','surprise']

                emotion_prob_list=[0,0,0,1,0,0]
                emotion_class_dic={}
                for i in range(6):
                    emotion_class_dic[emotion_class_list[i]]=emotion_prob_list[i]

            if(saved=="0"):
                timeItem=str(datetime.datetime.strftime(timenow, '%Y-%m-%d %H:%M:%S'))
                data_mongo = {'userName':userName,'use_date':timeItem}
                # selfEmo = [userName, str(timeItem)]
                for item in emotion_class_dic:
                    if(item!='neutral'):
                        # selfEmo.append(emotion_class_dic[item]*100)
                        data_mongo[item] = emotion_class_dic[item]*100
                        # print(selfEmo)
                        print(data_mongo)


                current_collection = current_db[collection_name]
                result = current_collection.insert_one(data_mongo)
                print(result)

                lis = [dict(userName=current_data['userName'], use_date=current_data['use_date'],
                            angry=current_data['angry'], sad=current_data['sad'], fear=current_data['fear'],
                            happy=current_data['happy'], surprise=current_data['surprise']) for current_data in
                       current_collection.find()]
                print(lis)


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
