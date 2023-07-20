import numpy as np
import streamlit as st
import cv2
from keras.models import load_model
#loading the model
model=load_model('dog_breed.h5')
classnames=['scottish_deerhound','maltese_dog','bernese_mountain_dog']
#settingb title of app
st.title("Dog Breed Prediction")
st.markdown("Upload image of the dog")
#uploadung the image of dog
dog_image=st.file_uploader("Choose an image...",type='png')
submit=st.button("Predict")
if submit:
	if dog_image is not None:
		#converting the file into an opencv image
		file_bytes = np.asarray(bytearray(dog_image.read()), dtype=np.uint8)
		opencv_image=cv2.imdecode(file_bytes,1)
		#displaying the image
		st.image(opencv_image,channels="BGR")
		#resize
		opencv_image=cv2.resize(opencv_image,(224,224))
		#convert image intp 4 dimension
		opencv_image.shape=(1,224,224,3)
		#predict
		y_pred=model.predict(opencv_image)
		st.title(str("The Dog Breed is "+classnames[np.argmax(y_pred)]))