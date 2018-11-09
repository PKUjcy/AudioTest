from pytube import YouTube
import pprint as pprint
import numpy as np
import urllib.request
from venv.get_video import get_video
import urllib.error
import json as json
import csv as csv
import tensorflow as tf
import numpy as np
import os
from pandas import read_csv
"""
文件介绍：
    bal_train 文件夹：          用于保存tensorflow形式的音频提取特征
    ontology.json:              用于保存整个dataset包含的音频的不同种类的id，介绍
    class_labels_indices.csv    用于保存tensorflow形式的文件音频提取特征label所对应的音频分类标号
    balanced_train_segments.csv 用于保存平衡的音频数据集的youtube视频ID,类ID，时间点
"""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#music_label_dic 用于存储属于音乐这个范畴下的种类ID和对应的信息

class read_massage(object):
    _music_label_dic={}
    _music_label=[]
    _all_label_dic={}
    #audio_feature 用于存储每个音频的特征属性与对应的种类ID
    _audio_feature=[]
    _music_audio_feature = {}
    _no_music_audio_feature = {}
    def read_label_index(self):
        label_index=read_csv('class_labels_indices.csv')
        #print(label_index)
        #print(type(label_index))
        #print(label_index['display_name'][3])
        return label_index
    def read_all_message(self):
        filename = 'balanced_train_segments.csv'
        with open(filename) as f:
            train_segments_reader = csv.reader(f)
        self.read_label_index()
        with open("ontology.json",'r') as load_f:
            ontology = json.load(load_f)
