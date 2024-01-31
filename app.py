import urllib.request
import json
import os
import ssl

def allowSelfSignedHttps(allowed):
    # bypass the server certificate verification on client side
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context
 
allowSelfSignedHttps(True) # this line is needed if you use self-signed certificate in your scoring service.
 
 
def model_predict(gender,age, hypertension,heart_disease, ever_married, work_type,Residence_type, avg_glucose_level, bmi, smoking_status):
    # Request data goes here
    data = {
        "Inputs": {
            "input1":
            [
                {
                    "gender": gender,
                    "age":age,
                    "hypertension": hypertension,
                    "heart_disease": heart_disease,
                    "ever_married": ever_married,
                    "work_type": work_type,
                    "Residence_type": Residence_type,
                    "avg_glucose_level": avg_glucose_level, 
                    "bmi": bmi,
                    "smoking_status": smoking_status
                }
                
            ]
        },
    }
 
    body = str.encode(json.dumps(data))
 
    url = 'http://6494f16b-4f76-4f40-b28a-c50f43a52925.eastus.azurecontainer.io/score'
    api_key = 'tJIDfHOTrcH2AXmTpBT5ZC9MyVMvSiz8' # Replace this with the API key for the web service
    headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}
 
    req = urllib.request.Request(url, body, headers)
 
    try:
        response = urllib.request.urlopen(req)
        result = response.read()
        result_decode = result.decode('utf-8')
        return json.loads(result_decode)
    except urllib.error.HTTPError as error:
        print("The request failed with status code: " + str(error.code))
        # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
        print(error.info())
        print(error.read().decode("utf8", 'ignore'))

from flask import Flask, request, render_template
app = Flask(__name__)

@app.route("/")
def formPage():
    return render_template('form.html')
 
@app.route("/submit")
def submit():
    if request.method == 'POST':
        form_data = request.form
        return render_template('form.html',prediction = 123)
 
if __name__ == "__main__":
    app.run()