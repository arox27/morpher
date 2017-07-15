from flask import render_template, request
from werkzeug import secure_filename
from morpher import morpher_main
from app import app

@app.route('/')
@app.route('/index')
def index():
    user = {'nickname': 'Miguel'}  # fake user
    posts = [  # fake array of posts
        {
            'author': {'nickname': 'John'},
            'body': 'Beautiful day in Portland!'
        },
        {
            'author': {'nickname': 'Susan'},
            'body': 'The Avengers movie was so cool!'
        }
    ]
    return render_template("index.html",
                           title='Home',
                           user=user,
                           posts=posts)

@app.route('/upload')
def form():
   return render_template("form.html")

@app.route('/morpher', methods = ['GET', 'POST'])
def morph():
        if request.method == 'POST':
                f1 = request.files.get('img1', '')
                f2 = request.files.get('img2', '')
                f1.save(secure_filename(f1.filename))
                f2.save(secure_filename(f2.filename))
                morpher_main(f1.filename, f2.filename, "new.gif")
                return "morphing..."
