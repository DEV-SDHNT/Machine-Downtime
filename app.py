import os
import numpy as np
import pandas as pd
import joblib as jlib
import json
from flask import Flask,render_template,request,jsonify,Response
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix


app=Flask(__name__)

app.config['UPLOAD_FOLDER']='uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'],exist_ok=True)

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/upload",methods=['POST'])
def trainer():
    if 'file' not in request.files:
        return "No file in the request",400
    file=request.files['file']
    if file.filename=='':
        return "No file Selected ! ",400
    if file and file.filename.endswith('.csv'):
        filepath=os.path.join(app.config['UPLOAD_FOLDER'],file.filename)
        file.save(filepath)
        print(filepath)
        try:
            ds=pd.read_csv(filepath,usecols=['Hydraulic_Pressure(bar)','Coolant_Temperature','Downtime'])
            ds=ds.dropna()
            print(ds)
            # exclude=['Date','Machine_ID','Assembly_Line_No','Downtime']
            # x=ds.drop(columns=exclude)
            x=ds[['Hydraulic_Pressure(bar)','Coolant_Temperature']]
            y=ds['Downtime']
            print(x.shape)
            xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=43)

            model=RandomForestClassifier(n_jobs=-1,verbose=True)
            model.fit(xtrain,ytrain)
            jlib.dump(model,'MacDown.pkl')        
            modelpred=model.predict(xtest)

            print("Model Accuracy ::",accuracy_score(ytest,modelpred))
            print("Confusion Matrix ::\n ",confusion_matrix(ytest,modelpred))
            print("Report :: \n",classification_report(ytest,modelpred))
            return render_template('machineDowntime.html')
        except Exception as e:
            return f'Error ! :: {e}',500
    else:
        return 'Invalid file format',400

@app.route('/modeltest',methods=['POST'])
def Model():
    # CoolantTemperature=request.form.get('CoolantTemperature')
    # HydraulicPressure=request.form.get('HydraulicPressure')
    print("Model testing")
    userinput=request.get_json();
    print('Request is json::',request.is_json)
    HydraulicPressure=userinput.get('HydraulicPressure')
    CoolantTemperature=userinput.get('CoolantTemperature')
    test=[HydraulicPressure,CoolantTemperature]
    print(test)
    test=np.array(test)
    test.reshape(1,-1)
    model=jlib.load('MacDown.pkl')
    prediction=model.predict(test.reshape(1,-1))
    prediction=prediction[0]
    result={'Downtime ':f'{prediction}'}
    if request.is_json:
        print(result)
        return jsonify(result)
    else:
        return jsonify({"error":'Request not json'}),400

if __name__=='__main__':
    app.run(debug=True)