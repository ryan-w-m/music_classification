from __future__ import unicode_literals

import pandas as pd
import numpy as np
from math import sqrt
import os
import csv

#Video
import youtube_dl
from youtube_dl import YoutubeDL

# Preprocessing
import librosa
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import LabelEncoder, StandardScaler
#from sklearn.neighbors import KNeighborsClassifier

#Keras
import keras
#from keras import models
#from keras import layers
from keras.models import load_model

#importing model and reading data used for model
data = pd.read_csv('data.csv')
data = data.drop(['filename'],axis=1)
model = load_model('my_model.h5')


##Writing fuctions to standardise the dataset. 'means' and 'stdevs' will be used to standardise user data.
X_new = np.array(data.iloc[:, :-1], dtype = float)

# calculate column means
def column_means(dataset):
    means = [0 for i in range(len(dataset[0]))]
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        means[i] = sum(col_values) / float(len(dataset))
        means = np.array(means)
    return means

means = column_means(X_new)

# calculate column standard deviations
def column_stdevs(dataset, means):
    stdevs = [0 for i in range(len(dataset[0]))]
    for i in range(len(dataset[0])):
        variance = [pow(row[i]-means[i], 2) for row in dataset]
        stdevs[i] = sum(variance)
    stdevs = [sqrt(x/(float(len(dataset)-1))) for x in stdevs]
    return stdevs

stdevs = column_stdevs(X_new, means)

"""
#standadize the dataset
def standardise_dataset(dataset, means, stdevs):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - means[i]) / stdevs[i]

standardise_dataset(X_new, means, stdevs)
"""
##accepting a URL as input, downloading the video to a desired directory and converting to a WAV file
user_url = str(input("Please enter a Youtube URL for the song of your choice: "))

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

ydl_opts = {
    'format': 'bestaudio/best',
    'outtmpl': '/home/ryan/genres/genre_classifier/test_data/%(title)s.%(ext)s',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'wav',
        'preferredquality': '192',
    }],
    'logger': MyLogger(),
    'progress_hooks': [my_hook],
}
with youtube_dl.YoutubeDL(ydl_opts) as ydl:
    ydl.download([user_url])
    
#removing spaces in the user file name  
directory = '/home/ryan/genres/genre_classifier/test_data'
[os.rename(os.path.join(directory, f), os.path.join(directory, f).replace(' ', '_').lower()) for f in os.listdir(directory)]

#writing a header for the user data
header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
for i in range(1, 21):
    header += f' mfcc{i}'
header += ' label'
header = header.split()

#creating the features of the user file in the same way as for the original dataset
file = open('user_data.csv', 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)
for filename in os.listdir('test_data'):
    songname = f'./test_data/{filename}'
    print(songname)
    y, sr = librosa.load(songname, mono=True, duration=30)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rms)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
    for e in mfcc:
        to_append += f' {np.mean(e)}'
    file = open('user_data.csv', 'a', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(to_append.split())
        
#reading and standardising the user data (one observation)
user_data = pd.read_csv('user_data.csv')
user_data = user_data.drop(['filename'],axis=1)
X_user = user_data.drop(labels='label', axis = 1)
X_user = np.array(X_user)

X_user_scaled = (X_user - means)/stdevs

#Predicting on the user data using Convolution Neural Network
user_pred = model.predict(X_user_scaled)
output = np.argmax(user_pred)

this_dict = {
    0 : 'blues',
    1 : 'classical',
    2 : 'country',
    3 : 'disco',
    4 : 'hiphop',
    5 : 'jazz',
    6 : 'metal',
    7 : 'pop',
    8 : 'reggae',
    9 : 'rock' 
}

print("This song should fit into the " + str(this_dict[output]) + " genre.")

#Removing the user file from the directory 
dir = '/home/ryan/genres/genre_classifier/test_data'
for f in os.listdir(dir):
    os.remove(os.path.join(dir, f))
