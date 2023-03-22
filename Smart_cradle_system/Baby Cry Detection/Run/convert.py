import tensorflow as tf
from keras.models import model_from_json
with open('D:\BabyCryingDetection_raspberry\ModelWeights\cnn2.json', 'r') as f:
    mymodel=model_from_json(f.read())

mymodel.load_weights("D:\BabyCryingDetection_raspberry\ModelWeights\cnn2.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(mymodel)
tflite_model = converter.convert()
open('D:\BabyCryingDetection_raspberry\Raspberry pi  application/model.tflite','wb').write(tflite_model)