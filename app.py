import streamlit as st
import pandas as pd
import numpy as np
from prediction import predict

st.title('Classifying Potenitally Hazardous Nearest Earth Object')
st.markdown('Try this model to find whether the object is hazardous or not')

st.header('Object features')
col1, col2 = st.columns(2)
with col1:
    est_diameter_min = st.slider('minimum diameter (km)', 0.000609, 37.892650,1.0 )
    est_diameter_max = st.slider('maximum diameter (km)', 0.001362, 84.730541,1.0 )
    absolute_magnitude = st.slider('absolute magnitue(luminocity)', 9.23, 33.2,1.0 )
with col2:
    miss_distance = st.slider('miss distance (km)', 0.748, 675000.0,1.0 )
    relative_velocity = st.slider('relative velocity (km)', 204, 236990, 1)

if st.button('Predict Type of Object'):
    result = predict(np.array([[est_diameter_min, est_diameter_max, relative_velocity, miss_distance,absolute_magnitude]]))
    if result[0]==0:
        st.markdown(":green[Non-Hazardous Object]")
    elif result[0]==1:
        st.markdown(":red[Hazardous Object]")

        
st.text("")
st.text("")
st.text("This project is developed by")
st.text("Nagendra Mangali")
st.text("Hall TicketNo: 2206DSP133")