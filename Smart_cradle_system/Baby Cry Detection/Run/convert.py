import tensorflow as tf
from keras.models import model_from_json
with open('D:\\HK2-Năm 3\\PBL5\\Code\\Smart_cradle_system\\Model_self_data\\Model10\\cnn.json', 'r') as f:
    mymodel=model_from_json(f.read())

mymodel.load_weights("D:\\HK2-Năm 3\\PBL5\\Code\\Smart_cradle_system\\Model_self_data\\Model10\\cnn.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(mymodel)
tflite_model = converter.convert()
open('D:\\HK2-Năm 3\\PBL5\\Code\\Smart_cradle_system\\Model_self_data\\Model10\\model.tflite','wb').write(tflite_model)