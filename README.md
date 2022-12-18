# pneumonia_detection

* This is a deep-learning project for identifying chest X-rays of patients with pneumonia from those without it. It seggregates the X-Rays into 2 classes: "Pneumonia" and "Normal"
* The implementation is done using Pre-trained MobileNetV2 architecture as it gave higher accuracies on comparison and is a lightweight and faster model.The model been fine tuned to achieve a peak accuracy of 93.8 percent.
* The trained model has been saved into a python script and tensorflow dependency has been removed using a custom preprocess function similar as in keras image helper.
* The model is trained, validated and tested and then predictions and testing has been made in tflite compatible format.
* The project is containerized using docker and the image can be used via the Dockerfile provided

## File information
|File/Folder name|Description|
|---|--
|data/data_xray|Data set with train,validation and test data|
|Dockerfile|file to build and run the docker image in your system|
|MobileNetV2_v4_1_01_0.938.h5|trained model with 93.8 percent accuracy saved into .h5 format|
|notebook.ipynb|For all the preprocessing, EDA, model training,validation,adding layers,hyperparameter tuning.|
|lambda_func.py|function to make predictions|
|pneumoniadetector-model.tflite|tensorflow lite compatible model|
|test.py|file to test the model using custom image|
## How to run the pneumonia detection service on your system
* Make sure you have Ubuntu installed (you can use WSL)
* Add WSL extension on VS Code 
* Open the working directory forked from this repo in WSL window of VS Code
* You can directly run using ipython 3 in the interactive temrinal by using these commands:
     * ipython3
     * import lambda_func
     * lambda_func.predict('https://raw.githubusercontent.com/Sivapriyapj/pneumonia_detection/master/data/data_xray/test/PNEUMONIA/person100_bacteria_475')
* Using the terminal, pull and build the docker image using
     * docker build -t pneumonia_detector .
     * docker run -it --rm -p 8080:8080 pneumonia_detector:latest
     * Once the image is builts : python test.py
* If any errors persist, make sure you have the following installed:
     * ipython3
     * keras-image-helper
     * tflite_runtime
