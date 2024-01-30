import streamlit as st
import pickle
import numpy as np
import pandas as pd

#importing the model
forest = pickle.load(open("forest_model.pkl", 'rb'))
laptop_data = pd.read_pickle("laptop_data.pkl")

st.title("Laptop Price Predictor")

#brand
company = st.selectbox("Laptop Brand", laptop_data["Company"].unique())

#type of laptop
type = st.selectbox("Laptop Type", laptop_data["TypeName"].unique())

#Ram
ram = st.selectbox("RAM(in GB)", [2,4,6,8,12,16,24,32,64])

#weight
weight = st.number_input("Weight of the Laptop")

#Touchscreen
touchscreen = st.selectbox("Touchscreen", ["Yes", "No"])

#IPS
ips = st.selectbox("IPS", ["Yes", "No"])

#screen size
screen_size = st.number_input('Screen Size')

# resolution
resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

#CPU
cpu = st.selectbox("CPU", laptop_data["Cpu_Brand"].unique())

#HDD
hdd = st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])

#SSD
ssd = st.selectbox('SSD(in GB)', [0,8,128,256,512,1024])

#GPU
gpu = st.selectbox("GPU", laptop_data["Gpu_Brand"].unique())

#operating_system
os = st.selectbox("OS", laptop_data["Operating_Systems"].unique())

if st.button("Predict Price"):

    ppi = None
    if touchscreen == "Yes":
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == "Yes":
        ips = 1
    else:
        ips = 0

    X_res = int(resolution.split("x")[0])
    Y_res = int(resolution.split("x")[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size
    query = np.array([company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])

    query = query.reshape(1, 12)
    st.title("The predicted price of this configuration is " + str(int(np.exp(forest.predict(query)[0]))))
