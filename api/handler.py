import pickle
from flask import Flask, request, Response
from rossmann.Rossmann import Rossmann
import pandas as pd

# loading model
model = pickle.load(open(
    '/home/sildolfoneto/Documents/repos/tcc/rossmann-stores-sales/models/model_final/model_rossmann.pkl', 'rb') )

# Inicializa a API
app = Flask(__name__)

@app.route('/rossmann/predict', methods=['POST'])
def rossmann_predict():
    test_json = request.get_json()
    
    if test_json:
        if isinstance(test_json, dict):
            test_raw = pd.DataFrame(test_json, index=[0])
        else: #multiple example
            test_raw = pd.DataFrame(test_json, columns=test_json[0].keys())
    
        pipeline = Rossmann()
        
        df1 = pipeline.data_cleaning(test_raw)
        df2 = pipeline.feature_engineering(df1)
        df3 = pipeline.data_preparation(df2)
        df_response = pipeline.get_prediction(model, test_raw, df3)
        
        return df_response
    else: 
        return Response('{}', status=200, mimetype='application/json')


if __name__ == '__main__':
    app.run('127.0.0.1')