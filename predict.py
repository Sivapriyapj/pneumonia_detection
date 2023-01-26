### test dataset
#(df_test.iloc[0]).to_dict()
from flask import Flask,request,jsonify




input_file = 'MobileNetV2_v4_1_01_0.938'
f_in = open(input_file,'rb')
df_test,y_test,dv,model_Final = tflite_runtime.load(f_in)
f_in.close()




classes = {
        "NORMAL",
        "PNEUMONIA"  

}


app = Flask('Pneumonia predictor')
@app.route('/predict', methods=['POST'])

def predict():
    patient = request.get_json()

    X_value = preprocessor([patient])
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    pred= interpreter.get_tensor(output_index)
    float_predictions = pred[0].tolist()
    

    result = dict(zip(classes,float_predictions))
    return jsonify(result)


if __name__ == "__main__":   
    app.run(debug=True, host= '0.0.0.0', port=9696)