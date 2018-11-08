from __future__ import unicode_literals
from pytube import YouTube
import pprint as pprint
import numpy as np
import urllib.request

import urllib.error
import json as json
import csv as csv
import tensorflow as tf
import numpy as np
import os
import youtube_dl
from pandas import read_csv
from pandas import read_csv
#def download_video(url,path):
#     proxy_support = urllib.request.ProxyHandler(proxy)
#     opener = urllib.request.build_opener(proxy_support)
#     urllib.request.install_opener(opener)
#     global savepath
#     yt = YouTube(url)
#     print(yt.title)
#     stream = yt.streams.filter(only_audio=True,subtype = 'mp4').first()
#     print(stream)
#     stream.download(path)
"""
文件介绍：
    bal_train 文件夹：          用于保存tensorflow形式的音频提取特征
    ontology.json:              用于保存整个dataset包含的音频的不同种类的id，介绍
    class_labels_indices.csv    用于保存tensorflow形式的文件音频提取特征label所对应的音频分类标号
    balanced_train_segments.csv 用于保存
"""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

proxy = {'http':'http://127.0.0.1:1080',
         'https':'https://127.0.0.1:1080'}
class init_message(object):
    label_vid_id_index={}# 用于映射tfrecord数据里label与vid_id的关系
    music_label_dic={}  #music_label_dic 用于存储属于音乐这个范畴下的种类ID和对应的信息
    music_label=[]
    all_label_dic={}  #all_label_dic={} 用于存储所有的音频种类的ID <key(类_ID):value(json)>
    audio_feature=[]  #audio_feature 用于存储每个音频的特征属性与对应的种类ID
    music_audio_feature = {} #用于存储tf格式下的每个音乐的音频特征(tensro张量list表示) <key(youtube_id):value([tensor])>
    no_music_audio_feature = {}#用于存储tf格式下的每个非音乐的音频特征(tensro张量list表示) <key(youtube_id):value([tensor])>
    audio_time_index={}  #用于存储每个音频在youtube视频里的起始和结束位置

    def read_label_index(self):
        """
        读取class_labels_indices.csv文件
        初始化label_vid_id_index字典
        """
        label_index=read_csv('class_labels_indices.csv')
        for i in range(len(label_index)):
            self.label_vid_id_index[label_index['index'][i]] = label_index['mid'][i]
        return label_index

    def read_all_message(self):
        """
        调用其他初始函数，对所有文件进行初始化
        """
        init_message.read_label_index(self)

        with open("ontology.json",'r') as load_f:
            ontology = json.load(load_f)
        init_message.init_all_label(self,ontology)

    def init_all_label(self,ontology):
        """
        用于将ontology文件所有的信息存放在all_label_dic字典中
        同时调用init_music_label
        """
        for i in range(len(ontology)):
            self.all_label_dic[ontology[i]['id']] = ontology[i]
        music_dic_first_element = self.all_label_dic['/m/04rlf']
        init_message.init_music_label(self,**music_dic_first_element)
        return  music_dic_first_element

    def init_music_label(self,**music_dic_ele):
        """
        用于将所有的属于音乐类别下的信息存放在music_dic_ele字典中
        """
       # print(type(music_dic_ele['id']))
        self.music_label_dic[music_dic_ele['id']] = music_dic_ele
       # print(music_dic_ele['id'],' ',music_dic_ele['name'],'\n')
        if  music_dic_ele.get('child_ids'):
            for i in range(len(music_dic_ele['child_ids'])):
                #music_label_dic[music_dic_ele['child_ids'][i]] = all_label_dic[music_dic_ele['child_ids'][i]]
                init_message.init_music_label(self,**self.all_label_dic[music_dic_ele['child_ids'][i]])
        else:
            return
        return

    def init_balanced_train_segments(self):
        filename = 'balanced_train_segments.csv'
        with open(filename) as f:
            train_segments_reader = csv.reader(f)

        return

class read_tfrecord_feature(object):
    file_message = init_message()
    def __init__(self,init_message_ele):
        file_message = init_message_ele
    def get_audio_label_feature(self,labels,vid_ids,audio_frame):
        self.file_message.audio_feature.append(audio_frame)
        flage = 0
        for i in range(len(labels)):
            if( self.file_message.music_label_dic.get( self.file_message.label_vid_id_index.get(labels[i]))!= None):
                flage = 1
                break
        if flage == 1:
            print('music_audio: '+'  '+vid_ids)
            self.file_message.music_audio_feature[vid_ids] = audio_frame
        else :
            print('no_music_audio: '+'  '+vid_ids)
            self.file_message.no_music_audio_feature[vid_ids] = audio_frame
    def extract_audio_feature(self,file_path):
        audio_record =file_path
        vid_ids = []
        labels = []
        audio_embedding = []
        start_time_seconds = [] # in secondes
        end_time_seconds = []
        feat_audio = []
        for example in tf.python_io.tf_record_iterator(audio_record):
            tf_example = tf.train.Example.FromString(example)
           # print(tf_example)
            vid_id = tf_example.features.feature['video_id'].bytes_list.value[0].decode(encoding='UTF-8')
            label = tf_example.features.feature['labels'].int64_list.value
            start_time_seconds.append(tf_example.features.feature['start_time_seconds'].float_list.value)
            end_time_seconds.append(tf_example.features.feature['end_time_seconds'].float_list.value)
            tf_seq_example = tf.train.SequenceExample.FromString(example)
            n_frames = len(tf_seq_example.feature_lists.feature_list['audio_embedding'].feature)
            sess = tf.InteractiveSession()
            rgb_frame = []
            audio_frame = []
            # iterate through frames
            for i in range(n_frames):
                audio_frame.append(tf.reshape(tf.decode_raw(
                    tf_seq_example.feature_lists.feature_list['audio_embedding'].feature[i].bytes_list.value[0],tf.uint8),[1]) )
            sess.close()
            # with tf.Session() as sess:
            #     init_op = tf.global_variables_initializer()
            #     sess.run(init_op)
            #     for i in range(len(audio_frame)):
            #         print(audio_frame[i].eval())
            read_tfrecord_feature.get_audio_label_feature(self,label,vid_id,audio_frame)
        return feat_audio



class MyLogger(object):
    def debug(self, msg):
        pass
    def warning(self, msg):
        pass
    def error(self, msg):
        print(msg)
def my_hook(d):
    if d['status'] == 'finished':
        print('Done downloading, now converting ...')
class download_audio(object):

     def download_vidio(video_id,out_path):
        output = out_path+'/%(title)s.%(ext)s'
        path = 'https://www.youtube.com/watch?v='+video_id
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',

            }],
            'logger': MyLogger(),
            'progress_hooks': [my_hook],
            'outtmpl': output,
        }
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([path])

if __name__ == '__main__':
    rootdir = 'bal_train'
    t= init_message()
    t.read_all_message()
    list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
    v = read_tfrecord_feature(t)
    for i in range(0,len(list)):
        path = os.path.join(rootdir,list[i])
        if os.path.isfile(path):
            v.extract_audio_feature(path)

