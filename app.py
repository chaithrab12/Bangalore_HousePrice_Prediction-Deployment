import pickle

import numpy as np
from flask import Flask, render_template, request
import pandas as pd
from scipy.special import inv_boxcox
priceLambda =-0.3072523138741146
app = Flask(__name__)
data = pd.read_csv("Clean_data.csv")
columns = pd.read_csv("data_Location_Column.csv")
model = pickle.load(open("HouseDataPrediction.pkl", "rb"))
locations = sorted(columns['location'].unique())
data.drop(data.columns[0], axis=1, inplace=True)


@app.route('/')
def home():
    return render_template('index.html', locations=locations, result='')


@app.route('/predict', methods=['POST'])
def predict():
    print(request.form.get('location'))
    print(data['1st Phase JP Nagar'])

    loc_index = np.where(data.columns == request.form.get('location'))[0][0]

    x = np.zeros(len(data.columns))
    print(len(data.columns))
    x[0] = request.form.get('bath')
    x[1] = request.form.get('balcony')
    x[2] = request.form.get('bhk')
    x[3] = request.form.get('sqft')
    if loc_index >= 0:
        x[loc_index] = 1
    result = model.predict([x])[0]
    print(result)
    result = inv_boxcox(result, priceLambda)
    # return render_template("result.html", result)
    return render_template('index.html', locations=locations,
                           result="Price per sqft in {0} is {1}".format(request.form.get('location'),
                                                                             (result * 100000) / float(request.form.get(
                                                                                 'sqft'))))


if __name__ == "__main__":
    app.run(host='0.0.0.0',port=8080)
