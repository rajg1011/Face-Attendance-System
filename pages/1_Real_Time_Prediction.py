import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import time



# import the data from db
# import face_rec  # dont import it here,import at home so that redis connection will be join as it load and not much delay happen
# instead do this
from Home import face_rec
st.subheader("Real Time Prediction")
# retrieve the data from redis
with st.spinner("Loading Models and connecting to Redis DB..."):
    redis_face_db=face_rec.retrieve_data_from_redis(name='Raj:Project')
    st.dataframe(redis_face_db)

st.success("Data Successfully retrieved from database")


#Real time prediction

waitTime= 30  #time in seconds
setTime= time.time() # curr time
realtimePred=face_rec.RealTimePred()



#whole fucntion copied from github of wertc
def video_frame_callback(frame):
    global setTime  # we are use setIie defined above
    img = frame.to_ndarray(format="bgr24")  #bgfr is a 3d np array
    #operation that u can perform on array (image is converted to array above)
    pred_img= realtimePred.face_prediction(img,redis_face_db,'Facial_Features',thresh=0.5)
    timenow= time.time()
    difftime= timenow-setTime
    if difftime>=waitTime:
        realtimePred.save_logs_redis()
        setTime=time.time()


    return av.VideoFrame.from_ndarray(pred_img, format="bgr24")


webrtc_streamer(key="realtimePrediction", video_frame_callback=video_frame_callback,
rtc_configuration={
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)
