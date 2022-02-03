# load libraries
import pandas as pd
from fastai.text.all import *
from transformers import *
from blurr.data.all import *
from blurr.modeling.all import *
from flask import Flask,jsonify,request,send_file
from dotenv import load_dotenv
import os

# load facebook BART model
inf_learn = load_learner(fname='models/BART_Finetuned_CurationCorp.pkl')

# Define functions       
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
# set env for secret key
load_dotenv()

secret_id = os.getenv('AI_SERVICE_SECRET_KEY')

# print(secret_id)
def check_for_secret_id(request_data):    
    try:
        if 'secret_id' not in request_data.keys():
            return False, "Secret Key Not Found."
        
        else:
            if request_data['secret_id'] == secret_id:
                return True, "Secret Key Matched"
            else:
                return False, "Secret Key Does Not Match. Incorrect Key."
    except Exception as e:
        message = "Error while checking secret id: " + str(e)
        return False,message

@app.route('/summarizer',methods=['POST'])  #main function
def main():
    params = request.get_json()
    input_query=params["data"]
    input_query = input_query[0]['summary']
    key = params['secret_id']

    request_data = {'secret_id' : key}
    secret_id_status,secret_id_message = check_for_secret_id(request_data)
    print ("Secret ID Check: ", secret_id_status,secret_id_message)
    if not secret_id_status:
        return jsonify({'message':"Secret Key Does Not Match. Incorrect Key.",
                        'success':False}) 
    else:
        result_summary = inf_learn.blurr_generate(text)[0] #  generate summary
        output_json = {'summary: ':result_summary}
    return jsonify(output_json)     

if __name__ == '__main__':
    app.run()             