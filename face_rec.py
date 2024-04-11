import numpy as np
import streamlit as st
import pandas as pd
import cv2
import os
import redis
from insightface.app import FaceAnalysis
from sklearn.metrics import pairwise
import time #python inbuilt library no need to install
from datetime import datetime

#Connect to redis

#host=os.getenv('HOST')
host=st.secrets('HOST')
# port=os.getenv('PORT')
port=st.secrets('PORT')
# password=os.getenv('PASSWORD')
password=st.secrets('PASSWORD')
r = redis.StrictRedis(host=host, port=port, password=password)  #decode_responses=False
#configure face analysis
faceapp= FaceAnalysis(name='buffalo_sc',
                     root='insightFace_model',
                     providers=['CPUExecutionProvider'])
faceapp.prepare(ctx_id=0,det_size=(640,640),det_thresh=0.5)

#Retreive the data from Redis
def retrieve_data_from_redis(name):
    retrieve_dict = r.hgetall(name)
    retrieve_series = pd.Series(retrieve_dict) 
    retrieve_series = retrieve_series.apply(lambda x: np.frombuffer(x,dtype=np.float32)) 
    index = retrieve_series.index 
    index = list(map(lambda x: x.decode(),index)) 
    retrieve_series.index = index
    retrieve_frame = retrieve_series.to_frame().reset_index() #reset_index will provide index
    retrieve_frame.columns=['name_role','Facial_Features']
    retrieve_frame[['Name','Role']]=retrieve_frame['name_role'].apply(lambda x: x.split('@')).apply(pd.Series)
    return retrieve_frame[['Name','Role','Facial_Features']]



#ML Search Algo
def ml_search_algo(dataframe, feature_column_name, test_vector, thresh=0.5):
    dataframe=dataframe.copy()
    X_list= dataframe[feature_column_name].tolist()
    x=np.asarray(X_list)
    #step3: cal. cosine
    similar= pairwise.cosine_similarity(x,test_vector.reshape(1,-1))
    similar_arr = np.array(similar).flatten() 
    dataframe['cosine']=similar_arr

    data_filter=dataframe.query(f'cosine>={thresh}')
    if len(data_filter)>0:
        data_filter.reset_index(drop=True,inplace=True)
        argmax=data_filter['cosine'].argmax()
        name, role= data_filter.loc[argmax][['Name','Role']]
    else:
        name='Unknown'
        role='Unknown'
    return name, role


class RealTimePred:
    def __int__(self):  #constructor function
        self.logs=dict(name=[],role=[],current_time=[])  #logs is just the name given and it create dictionary name logs
    def reset_dict(self):
        self.logs=dict(name=[],role=[],current_time=[])
    
    def save_logs_redis(self):
        #step1: create a logs dataframe
        dataframe = pd.DataFrame(self.logs) 
        #step2: drop the duplicate information as in one min, it will store too much time same name
        dataframe.drop_duplicates('name', inplace=True) #inplace means changes should be made in the same dataframe
        #step3: save data to redis (list)
        name_list=dataframe['name'].tolist()
        role_list=dataframe['role'].tolist()
        ctime_list=dataframe['current_time'].tolist()
        encoded_data=[]
        for name, role, ctime in zip(name_list,role_list,ctime_list): #zip means do all the things together and this give 
              #1 name_list, 1 role_list and 1 cite_list like- 'Raj' , "Student", 12:90
            if name!='Unknown':
                concat_string=f"{name}@{role}@{ctime}"
                encoded_data.append(concat_string)

        if len(encoded_data)>0:
           r.lpush('Raj:logs',*encoded_data)

        self.reset_dict()        

#Face Prediction
    def face_prediction(self,test_img, dataframe, feature_column_name, thresh=0.5):
        current_time= str(datetime.now()) # to get current time
        results=faceapp.get(test_img)
        test_copy=test_img.copy()
        for res in results:
            x1,y1,x2,y2=res['bbox'].astype(int)
            embeddings=res['embedding']
            name,role=ml_search_algo(dataframe,feature_column_name,test_vector=embeddings,thresh=thresh)
            if name=='Unknown':
                color=(0,0,255)
            else:
                color= (0,255,0)
            cv2.rectangle(test_copy,(x1,y1),(x2,y2),color)
            test_gen=name
            cv2.putText(test_copy,test_gen,(x1,y1-25),cv2.FONT_HERSHEY_DUPLEX,0.7,color,2)
            #put time on image
            cv2.putText(test_copy,current_time, (x1,y1-10),cv2.FONT_HERSHEY_DUPLEX,0.5,color,1)

            # save info in log dicts
            # self.logs['name'].append(name)
            # self.logs['role'].append(role)
            # self.logs['current_time'].append(current_time)

        return test_copy


#Registration form:

class RegistrationForm:
    def __init__(self):
        self.sample=0
    def reset(self):
        self.sample=0
    def get_embeddings(self,frame):
        result= faceapp.get(frame,max_num=1)
        embeddings=None #we write this line for case if no frame is detected then it may give eoor in return
        for res in result:
            self.sample+=1
            x1,y1,x2,y2=res['bbox'].astype(int)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),1)
            #put no. of sample taken in frame:
            text=f"samples = {self.sample}"
            cv2.putText(frame,text,(x1,y1),cv2.FONT_HERSHEY_DUPLEX,0.6,(255,255,0),2)
            embeddings=res['embedding']
        return frame,embeddings
    
    def save_data_redis(self,name , role):
        #validations
        if name is not None:
            if name.strip()!='':  #stipe remove spaces from front
                key=f'{name}@{role}'
            else:
                return 'name_false'
        else:
            return 'name_false'
        
        #validation on if file is available
        if 'face_embedding.txt' not in os.listdir():
            return 'file_not_found'

        # step:1 load face_embedding.txt file
        x_arr=np.loadtxt('face_embedding.txt', dtype=np.float32)  #flatten array
        #step2: convert into array ( proper hape (proper array))
        sample= int(x_arr.size/512)  #each face embedding have 512 values and if we divide it by 512  then we get no. of samples
        x_arr=x_arr.reshape(sample,512) # it will give in proper shape
        x_arr=np.asarray(x_arr) 
        #step3: cal. mean
        x_mean=x_arr.mean(axis=0) 
        x_mean=x_mean.astype(np.float32)
        x_mean_byte=x_mean.tobytes() 
        #step4: save in redis
        r.hset(name='Raj:Project', key=key, value=x_mean_byte)

        #remove file
        os.remove('face_embedding.txt')
        self.reset()

        return True



  
