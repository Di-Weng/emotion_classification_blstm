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
from pyecharts import ThemeRiver
#app.config.from_envvar('FLASKR_SETTINGS', silent=True)

# configuration

DATABASE = 'flaskr.db'
DEBUG = True
SECRET_KEY = 'development key'
USERNAME = 'admin'
PASSWORD = 'default'
emotion_model_path = 'model/best_model.h5'
gender_model_path = 'model/gender_model.h5'
# 加载第一个模型
g1 = tf.Graph() # 加载到Session 1的graph
g2 = tf.Graph() # 加载到Session 2的graph

sess1 = tf.Session(graph=g1) # Session1
sess2 = tf.Session(graph=g2) # Session2
with sess1.as_default():
    with g1.as_default():
        emotion_model = load_model(emotion_model_path)

# 加载第二个模型
with sess2.as_default():  # 1
    with g2.as_default():
        gender_model = load_model(gender_model_path)


app = Flask(__name__)
app.config.from_object(__name__)

def connect_db():
    return sqlite3.connect(app.config['DATABASE'])

def init_db():
    with closing(connect_db()) as db:
        with app.open_resource('schema.sql', mode='r') as f:
            db.cursor().executescript(f.read())
        db.commit()
#

# @app.route('/')
# def hello_world():
#     return 'Hello World!'


@app.before_request
def before_request():
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



@app.route('/emo_visual')
def emo_visual():
    data = [ ['2015/11/08', 10, 'angry'], ['2015/11/09', 15, 'angry'], ['2015/11/10', 35, 'angry'],['2015/11/08', 1, 'happy'], ['2015/11/09', 76, 'happy'], ['2015/11/10', 15, 'happy'],['2015/11/08', 1, 'surprise'], ['2015/11/09', 0, 'surprise'], ['2015/11/10', 20, 'surprise'],['2015/11/08', 0, 'sad'], ['2015/11/09', 5, 'sad'], ['2015/11/10', 0, 'sad'],['2015/11/08', 88, 'fear'], ['2015/11/09',4, 'fear'], ['2015/11/10', 30, 'fear'],]
    myechart = ThemeRiver("主题河流图",width = 800,height = 500)
    myechart.add(['angry', 'happy', 'surprise', 'sad', 'fear'], data, is_label_show=False,is_legend_show = True)
    return render_template('emo_visual.html',myechart = myechart.render_embed(),script_list = myechart.get_js_dependencies(),host = "https://pyecharts.github.io/assets/js")
    



# @app.route('/entries')
# def show_entries():
#     cur = g.db.execute('select title, text from entries order by id desc')
#     entries = [dict(title=row[0], text=row[1]) for row in cur.fetchall()]
#     return render_template('show_entries.html', entries=entries)

# @app.route('/add', methods=['POST'])
# def add_entry():
#     if not session.get('logged_in'):
#         abort(401)
#     g.db.execute('insert into entries (title, text) values (?, ?)',
#                  [request.form['title'], request.form['text']])
#     g.db.commit()
#     flash('New entry was successfully posted')
#     return redirect(url_for('show_entries'))


# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     error = None
#     if request.method == 'POST':
#         if request.form['username'] != app.config['USERNAME']:
#             error = 'Invalid username'
#         elif request.form['password'] != app.config['PASSWORD']:
#             error = 'Invalid password'
#         else:
#             session['logged_in'] = True
#             flash('You were logged in')
#             return redirect(url_for('show_entries'))
#     return render_template('login.html', error=error)

# @app.route('/logout')
# def logout():
#     session.pop('logged_in', None)
#     flash('You were logged out')
#     return redirect(url_for('show_entries'))

# @app.route('/test')
# def test():
#     return render_template('test.html')



@app.route('/get_audio', methods=['GET', 'POST'])
def get_audio():
        return render_template('get_audio.html')

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

            # emotion prediction
            with sess1.as_default():
                with sess1.graph.as_default():
                    emotion_predict_class, emotion_predict_prob, emotion_class_dic = get_audioclass(emotion_model, filename,
                                                                                                    'emotion', all=True)


            # gender prediction
            with sess2.as_default():
                with sess2.graph.as_default():
                    gender_predict_class, gender_predict_prob,gender_class_dic = get_audioclass(gender_model, filename, 'gender', all=True)

            jsonData = {}
            emotion_class_list = []
            emotion_prob_list = []
            gender_class_list = []
            gender_prob_list = []

            print(emotion_class_dic)



            for current_emotion_class, current_emotion_prob in emotion_class_dic.items():
                emotion_class_list.append(current_emotion_class)
                emotion_prob_list.append(float(current_emotion_prob))

            for current_gender_class, current_gender_prob in gender_class_dic.items():
                gender_class_list.append(current_gender_class)
                gender_prob_list.append(float(current_gender_prob))

            jsonData['emotion_class'] = emotion_class_list
            jsonData['emotion_prob'] = emotion_prob_list
            jsonData['gender_class'] = gender_class_list
            jsonData['gender_prob'] = gender_prob_list
            return_data = json.dumps(jsonData)
            print(return_data)
            return (return_data)
    # else:
    #     if(request.method=='POST'):
    #         filename=request.headers["filename"]
    #         with sess1.as_default():
    #             with sess1.graph.as_default():
    #                 emotion_predict_class, emotion_predict_prob, emotion_class_dic = get_audioclass(emotion_model,
    #                                                                                                 filename,
    #                                                                                                 'emotion', all=True)
    #         # gender prediction
    #         with sess2.as_default():
    #             with sess2.graph.as_default():
    #                 gender_predict_class, gender_predict_prob = get_audioclass(gender_model, filename, 'gender',
    #                                                                            all=False)
    #         jsonData = {}
    #         emotion_class_list = []
    #         emotion_prob_list = []
    #         gender_class_list = []
    #         gender_prob_list = []
    #         print(emotion_class_dic)
    #         for current_emotion_class, current_emotion_prob in emotion_class_dic.items():
    #             emotion_class_list.append(current_emotion_class)
    #             emotion_prob_list.append(float(current_emotion_prob))
    #         for current_gender_class, current_gender_prob in emotion_class_dic.items():
    #             gender_class_list.append(current_gender_class)
    #             gender_prob_list.append(float(current_gender_prob))
    #         jsonData['emotion_class'] = emotion_class_list
    #         jsonData['emotion_prob'] = emotion_prob_list
    #         jsonData['gender_class'] = gender_class_list
    #         jsonData['gender_prob'] = gender_prob_list
    #         return_data = json.dumps(jsonData)
    #         return(return_data)

if __name__ == '__main__':
    app.run()

