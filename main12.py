from cgi import test
from this import d
from flask import Flask,request,jsonify
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

random_tune =pickle.load(open("random_tuned1.pkl","rb"))
columns_list=pickle.load(open("column_list1.pkl","rb"))

@app.route("/")
def home_page():
    return "welcome to my dashboard"

@app.route("/prediction")
def prediction():
    data =request.get_json()
    CRIM = data['CRIM']
    INDUS =data['INDUS']
    NOX =data['NOX']
    AGE = data['AGE']

    data_frame = {"CRIM":[CRIM],"INDUS":[INDUS],"NOX":[NOX],"AGE":[AGE]}

    test_data = pd.DataFrame(data_frame)

    prediction = random_tune.predict(test_data)

    return jsonify({"prediction":np.around(prediction[0],2),
                    "CRIM":CRIM,
                    "INDUS":INDUS,
                    "NOX":NOX,
                    "AGE":AGE})
if __name__ == "__main__" :
    app.run(host='0.0.0.0',port=5012)


                    