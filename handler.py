import pickle
import pandas as pd
from flask import Flask, request, Response
from rossmann.Rossmann import Rossmann
import os


# loading model
model = pickle.load(open('model/model_rossman.pkl', 'rb'))

# initialize API
app = Flask(__name__)

# Criar a rota, Endpoint. Receber a URL
@app.route('/rossmann/predict', methods=['POST']) # 'Post'= envia, 'Get' = recebe dados

def rossmann_predict():
    test_json = request.get_json()

    if test_json:
        if isinstance(test_json, dict): # unique example
            test_raw = pd.DataFrame(test_json, index=[0])

        else: # multiple examples
            test_raw = pd.DataFrame(test_json, columns=test_json[0].keys())

        # Instance Rossmann Class
        pipeline = Rossmann()

        # data cleaning
        df1 = pipeline.data_cleaning(test_raw)

        # feature engineering
        df2 = pipeline.feature_engineering(df1)

        # data preparation
        df3 = pipeline.data_preparation(df2)
        
        # prediction
        df_response = pipeline.get_prediction(model, test_raw, df3)

        return df_response

    else:
        Response('{}', status=200, mimetype='application/json')


if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)
    app.run(host='0.0.0.0', port=port) # IP localhost