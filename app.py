# all the imports
import sqlite3
from flask import Flask, request, session, g, redirect, url_for, \
     abort, render_template, flash
from contextlib import closing
import os
import datetime
import random
from predict import load_model,get_audioclass,analyse_emotionn
#app.config.from_envvar('FLASKR_SETTINGS', silent=True)

# configuration

DATABASE = 'flaskr.db'
DEBUG = True
SECRET_KEY = 'development key'
USERNAME = 'admin'
PASSWORD = 'default'

app = Flask(__name__)
app.config.from_object(__name__)
emotion_model_path = 'model/best_model_2.h5'

# model trainning :)
gender_model_path = 'model/gender_model.h5'

emotion_model = load_model(emotion_model_path)
gender_model = load_model(gender_model_path)

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
def show_index():
    return render_template('layout.html',dic = {"angry":0,"sad":0,"happy":0,"fear":0,"surprise":0})

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
    if request.method == 'POST':
        timenow=datetime.datetime.now()
        filename="recordFiles/"+datetime.datetime.strftime(timenow,'%Y%m%d%H%M%S')+"_"+str(random.randint(1,10000))+".wav"
        request.files['audioData'].save(filename)
        # test_model(model_path,test_folder)

        # emotion class prediction
        # emotion_predict_class, emotion_predict_prob, emotion_class_dic = get_audioclass(emotion_model,filename,'emotion',all=True)

        # gender prediction
        # gender_predict_class, gender_predict_prob, gender_class_dic = get_audioclass(gender_model,filename,'gender',all=True)
        dic = analyse_emotionn(emotion_model,filename)
        return redirect(url_for('layout.html',dic =dic))
    else:
        return render_template('layout.html',dic = {"angry":0,"sad":0,"happy":0,"fear":0,"surprise":0})


if __name__ == '__main__':
    app.run()

