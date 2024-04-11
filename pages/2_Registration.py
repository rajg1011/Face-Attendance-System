import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import numpy as np
import cv2
from Home import face_rec

st.subheader("Registration")
registration_from= face_rec.RegistrationForm()

#Step:1 Collect Person name and role
person_name= st.text_input(label="Name", placeholder="Your Name")
role= st.selectbox(label="Select Role",options=('Student', 'Teacher'))

#Step:2 Collect facial embedding
def video_callback_func(frame):  # from webrtc
    img = frame.to_ndarray(format="bgr24")
    reg_image, embeddings=registration_from.get_embeddings(img)  # to take the embeddings of the image
    # we can not directly save embeddings from here to redis, as it is a callback function and we cant stop it to save my data
    #So what we can do is we save it into local computer (.txt file as we have to append data in every loop of sample collect)
    # and then we put in redis after this function
    if embeddings is not None:  # function to save file in loca computer
        with open('face_embedding.txt',mode='ab') as f: #mode as ab means append in byte. also have value like r(read), w(write)
            np.savetxt(f,embeddings)

    return av.VideoFrame.from_ndarray(reg_image, format="bgr24")  #it return that image array form


webrtc_streamer(key="registration", video_frame_callback=video_callback_func.
rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)  #webrtc_streamer is used to start a WebRTC streaming session

#Step3: Save data in redis
if st.button("Submit"):
    return_val=registration_from.save_data_redis(person_name,role)
    if return_val== True:
        st.success(f"{person_name} registered successfully")
    elif return_val=='name_false':
        st.error("Please enter a valid name")
    elif return_val=='file_not_found':
        st.error("Image Capture file not found. Please refresh")




