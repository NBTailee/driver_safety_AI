import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from ultralytics import YOLO
import torch


if "conf" not in st.session_state:
    st.session_state["conf"] = 0
if "detect" not in st.session_state:
    st.session_state["detect"] = False

st.title("Seatbelt & Sleepy Driver Detection")

st.header("Detection Type")
option = st.selectbox("Choose Detection Type", ("drowsy","seatbelt"), index=None, placeholder="Select contact method...")
st.write("Type Chosen: ", option)

@st.cache_data
def load_model(type):
    if(type == "drowsy"):
        _model = YOLO(r"drowsy_driver\model\drowsy.pt")
    elif(type == "seatbelt"):
        _model = torch.hub.load(r"ultralytics/yolov5", "custom", path=r"drowsy_driver\model\seatbelt.pt", force_reload=True)
    return _model

st.subheader("Loading the model")

if option == None:
    st.text("You need to choose the model type first")
else:
    loading_model = st.text(f"Loading {option} model .....")
    model = load_model(option)
    loading_model.text("Loading Done!!!")

st.sidebar.header("Upload File")
if(option == None):
    st.sidebar.subheader("Choose a model before uploading file")
else:
    uploaded_file = st.sidebar.file_uploader("Choose a file", type=['jpeg','jpg'])
    if uploaded_file != None:
        st.subheader("Original Image:")
        file_type = uploaded_file.type.split("/")
        if(file_type[0] == "image"):
            st.image(uploaded_file, width=400)
        elif(file_type[0] == "video"):
            st.video(uploaded_file)
        st.header("Choosing Hyperparameter")
        st.subheader("F1-Curve")
        col1, col3, col2 = st.columns(3, gap="small")
        with col1:
            ori_img = Image.open(r"drowsy_driver\web\F1_curve_drowsy.jpg")
            st.image(ori_img,width=400 ,caption="Drowsy")
        with col2:
            ori_img = Image.open(r"drowsy_driver\web\F1_curve_seatbelt.jpg")
            st.image(ori_img,width=400 ,caption="Seatbelt")
        st.text("RECOMMEND Drowsy:0.4 & Seatbelt:0.4")
        st.slider("Choose Confidence Interval", 0.0,1.0, key="conf", value=0.4, step=0.1)
        btn = st.button("Detect", type="primary")
        if (btn):
            if option == "drowsy":
                if file_type[0] == "image":
                    img = Image.open(uploaded_file)
                    detecting = st.text("Detection processing.....")
                    result = model.predict(img, conf = st.session_state["conf"])
                    detecting.text("Detect Done!!!")
                    st.subheader("Detected Image:")
                    numpy_array = result[0].plot()
                    img = Image.fromarray(numpy_array[..., ::-1])
                    st.image(img, width=450, channels="RGB")
            elif option == "seatbelt":
                if file_type[0] == "image":
                    img = Image.open(uploaded_file)
                    detecting = st.text("Detection processing.....")
                    result = model(img)
                    model.conf = st.session_state["conf"]
                    detecting.text("Detect Done!!!")
                    result.show()
                    
                    

