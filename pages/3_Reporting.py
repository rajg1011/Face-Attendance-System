import streamlit as st #ML web dev lib
from Home import face_rec
st.set_page_config(page_title='Reporting',layout='wide')
st.subheader("Reporting")

#Retrieve logs from redis

name='Raj:logs'
def load_logs(name,end=-1):
    log_list= face_rec.r.lrange(name,start=0,end=end)
    return log_list


#make tab for we page
tab1,tab2=st.tabs(['Registered Data','Refresh Logs'])

with tab1:
    if st.button('Registered Data'):
        with st.spinner("Retrieving Data from Redis DB ..."):
            redis_face_db=face_rec.retrieve_data_from_redis(name='Raj:Project')
            st.dataframe(redis_face_db[['Name','Role']])

with tab2:
    if st.button('Refresh Logs'):  # give all the logs that are given attendance
        st.write(load_logs(name=name))


