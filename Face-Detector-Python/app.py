import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pytesseract as tess
import cv2
from PIL import Image
import os
import dlib
#face Detector
def faceFinder(img1):
    # Check if an image is uploaded
    if img1 is not None: 
        detector=dlib.get_frontal_face_detector()
        predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        gray=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        faces=detector(gray)
        for face in faces:
            landmarks=predictor(gray,face)
            x1,y1,x2,y2=face.left(),face.top(),face.right(),face.bottom()
            cv2.rectangle(img1,(x1,y1),(x2,y2),(0,255,0),2)
            for n in range(0,68):
                x=landmarks.part(n).x
                y=landmarks.part(n).y
                cv2.circle(img1,(255,0,0),-1)
        return img1
    else:
        # Handle the case where no image is uploaded
        st.write("Please upload an image.")  
        return None # Or a placeholder image if you prefer
#Text Extractor
def textExtractor(img2):
    #Check if an image is uploaded
    if img2 is not None:
        img2=Image.open(img2).convert("L")
        img2=cv2.medianBlur(np.array(img2),5)
        st.write("Extracting text...")
        return tess.image_to_string(img2)
        
    else:
        #Handle the case where no image is uploaded
        st.write("Please upload an image.")
        return None
def main():
    st.title("Face Detector")
    st.write("Upload an image to detect faces")
    img1=st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"],key="face_uploader")
    # Call faceFinder only if img1 is not None
    if img1 is not None: 
        res=faceFinder(img1)
        st.image(res, caption="Detected Faces", use_column_width=True)
    st.title("Text Extractor")
    st.write("Upload an image to extract text")
    img2=st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"],key="text_reader")
    #Call textExtractor only if img2 is not None
    if img2 is not None:
        result=textExtractor(img2)
        st.write("uploading Image...")
        st.write(result)
st.title("Face Detector and Text Extractor") 
main()