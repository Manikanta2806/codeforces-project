from flask import Flask,render_template,request
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import PolynomialFeatures

#create a Flask object
app = Flask(__name__) #Flask application
@app.route('/')
def hello():
    """test function"""
    return "Welcome to PVP College"
#app.run()

#First let's read the pickle file
with open('House.pkl','rb') as f:
    model = pickle.load(f)
@app.route('/',methods=['GET'])
def home():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
    Score = int(request.form['Score'])
    #now take the above form data and convert to array
    Poly_features=PolynomialFeatures(degree=3)
    pred = Poly_features.transform(np.array([[Score]]))
    prediction = model.predict(pred)
    #now we will pass above predicted data to template
    return render_template('index.html',
                           prediction = prediction)
app.run()

