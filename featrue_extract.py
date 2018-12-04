import glob
import os
import librosa
import numpy as np
import pydub

import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram

def windows(data, window_size):
    start = 0
    while start < len(data):
        yield start, start + window_size
        start += (window_size / 2)

def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)

    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=20).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    # mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    # contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    # tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
    #sr=sample_rate).T,axis=0)
    print(chroma.shape)
    return mfccs,chroma#,mel,contrast,tonnetz

def parse_audio_files(parent_dir,sub_dirs,file_ext="*.wav"):
    features, labels = np.empty((0,32)), np.empty(0)
    for label, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            try:
              mfccs, chroma = extract_feature(fn)
            except Exception as e:
              print ("Error encountered while parsing file: ", fn)
              continue
            ext_features = np.hstack([mfccs,chroma])
            features = np.vstack([features,ext_features])
            print(features.shape)
            if(sub_dir.find("music")!=-1):
                labels = np.append(labels, 1)
            else :
                labels = np.append(labels,0)
    return np.array(features), np.array(labels, dtype = np.int)

# def my_parse_audio_flies(parent_dir,sub_dirs,file_ext="*.wav",bands=64):
#     log_mel = np.array([])
#     labs = np.array([])
#     for label, sub_dir in enumerate(sub_dirs):
#         for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
#             audio ,sr  = librosa.load(fn)
#             if(sub_dir.find("music")!=-1):
#                 label = 1
#             else:
#                 label = 0
#             audio_mel = librosa.feature.mfcc(audio,sr,n_mfcc=20,
#                                              dct_type=2)
#             print(audio_mel.shape)
#
# my_parse_audio_flies(parent_dir="music_speech", sub_dirs =["music_wav","speech_wav"],bands=64)

# def parse_audio_files(parent_dir,sub_dirs,file_ext="*.wav",bands = 60, frames = 15):
#     window_size = 512 * (frames - 1)
#     log_specgrams = []
#     labels = []
#     for l, sub_dir in enumerate(sub_dirs):
#         for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
#             sound_clip,s = librosa.load(fn)
#             print(sound_clip.shape)
#             if(sub_dir.find("music")!=-1):
#                 label = 1
#             else:
#                 label = 0
#             for (start,end) in windows(sound_clip,window_size):
#               #(1)此处是为了是将大小不一样的音频文件用大小window_size，
#               #stride=window_size/2的窗口，分割为等大小的时间片段。
#               #(2)计算每一个分割片段的log mel_sepctrogram.
#               #或者，先分别计算大小不一的音频的log mel_spectrogram,在通过固定的窗口，
#               #切割等大小的频谱图。
#                 if(len(sound_clip[int(start):int(end)]) == window_size):
#                     signal = sound_clip[int(start):int(end)]
#
#                     melspec = librosa.feature.melspectrogram(signal,n_fft=2048,
#                                     hop_length=512,
#                                     n_mels = bands)
#                     print(melspec.shape)
#                     logspec = librosa.amplitude_to_db(melspec)
#                     logspec = logspec.T.flatten()[:, np.newaxis].T
#                     logspec = np.reshape(logspec,[bands*frames])
#
#                     log_specgrams.append(logspec)
#                     labels.append(label)
#     print(np.array(log_specgrams).shape)
#     return log_specgrams,labels

def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode

parent_dir = 'music_speech'
tr_sub_dirs = ["music_wav","speech_wav"]
ts_sub_dirs = ["test_music_wav","test_speech_wav"]
tr_features, tr_labels = parse_audio_files(parent_dir,tr_sub_dirs)
ts_features, ts_labels = parse_audio_files(parent_dir,ts_sub_dirs)
tr_labels = one_hot_encode(tr_labels)
ts_labels = one_hot_encode(ts_labels)
print(tr_labels)
