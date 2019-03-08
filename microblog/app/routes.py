from app import app
from flask import render_template, url_for
import pandas as pd
import random
import sqlite3

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/wine')
def wine():
    df = pd.read_csv('app/static/fake_wines.csv', sep='|')
    wine_ix = random.randint(0,len(df))
    return(str(df.iloc[wine_ix,:]['name']) + "==" + str(df.iloc[wine_ix,:]['description']))
    

@app.route('/wine_random')
def wine_random():
    exit
    filesize = 1500                 #size of the really big file
    offset = random.randrange(filesize)

    f = open('really_big_file')
    f.seek(offset)                  #go to random position
    f.readline()                    # discard - bound to be partial line
    random_line = f.readline()      # bingo!

    # extra to handle last/first line edge cases
    if len(random_line) == 0:       # we have hit the end
        f.seek(0)
        random_line = f.readline()  # so we'll grab the first line instead