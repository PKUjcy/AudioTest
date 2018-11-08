from pydub import AudioSegment
import  librosa
import os
from pandas import read_csv
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder
import librosa.display
import glob
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import  LogisticRegression
from sklearn.tree import  DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
###################音频分割测试代码 使用库 AudioSegment#######################
song = AudioSegment.from_wav("E:\BaiduNetdiskDownload\\test.wav")
##这里默认音频分割一帧是一毫秒
"""
song_segment_time = 10*1000

song_segment =[]
i=0
j=0
while i<song.duration_seconds*1000:
    song_segment.append(song[i:i+song_segment_time])
    i=i+1+song_segment_time
    j+=1

s=0
while(s<j):
    song_segment[s].export("song_segment_"+str(s),format="wav")
    s+=1
"""
##################Audio测试训练集###########################

#When you load the data, it gives you two objects; a numpy array of
# an audio file and the corresponding sampling rate by which it
# was extracted. Now to represent this as a waveform
# (which it originally is), use the following  code
data, sampling_rate = librosa.load('E:\\Urban_Sound_challenge_data\Train\Train\\2022.wav')##读取文件
plt.figure(figsize=(12, 4))
x_length = librosa.get_duration(data,sampling_rate)
print(x_length)
librosa.display.waveplot(data,sr=sampling_rate)
plt.show()
train = read_csv('E:\\Urban_Sound_challenge_data\Train\\train.csv')
def parser(row):
   # function to load files and extract features
   file_name = os.path.join(os.path.abspath('E:\\Urban_Sound_challenge_data\Train\\'), 'Train', str(row.ID) + '.wav')
   print(file_name)
   # handle exception to check if there isn't a file which is corrupted
   try:
      # here kaiser_fast is a technique used for faster extraction
      X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
      # we extract mfcc feature from data
      mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=30).T,axis=0)
   except Exception as e:
      print("Error encountered while parsing file: ", file_name)
      return None, None

   feature = mfccs
   label = row.Class

   return [feature, label]

temp = train.apply(parser, axis=1)
temp.columns = ['feature', 'label']
print("数据维度:行%s，列:%s"%train.shape)
print(train.head(20))
print(train.describe())
print(train.groupby('feature').size())
train.plot(kind='box',subplots = True, layout = (2,2),sharex = False,sharey = False)
pyplot.show()
train.hist()
pyplot.show()
scatter_matrix(train)

X = np.array(temp.feature.tolist())
y = np.array(temp.label.tolist())

lb = LabelEncoder()

#y = np_utils.to_categorical(lb.fit_transform(y))
