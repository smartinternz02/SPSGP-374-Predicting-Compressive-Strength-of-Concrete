import numpy as np
from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('cement.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html') 
@app.route('/Prediction',methods=['POST','GET'])
def prediction():
    return render_template('index1.html')
@app.route('/Home',methods=['POST','GET'])
def my_home():
    return render_template('home.html')
@app.route('/predict',methods=['POST'])

def index():
            
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    features_name = ['Cement', 'Blast Furnace Slag', 'Fly Ash', 'Water', 
                     'Superplasticizer','Coarse Aggregate', 'Fine Aggregate','Age']
    x = pd.DataFrame(features_value, columns=features_name)
    x_log=np.log(x)
    
    prediction=model.predict(x_log)
    print('prediction is', prediction)
    
    return render_template('result2.html',prediction_text=prediction)

if __name__ == "__main__":
    app.run(host = '127.0.0.1', port = 5000, debug = True)
    
    
