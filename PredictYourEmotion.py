# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 04:35:43 2023

@author: USER
"""

import streamlit as st
import librosa
import numpy as np
#from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import sounddevice as sd
import soundfile as sf

sample_rate = 48000


def feature_mfcc(
    waveform, 
    sample_rate,
    n_mfcc = 40,
    fft = 1024,
    winlen = 512,
    window='hamming',
    mels=128
    ):

    # Compute the MFCCs for all STFT frames 
    # 40 mel filterbanks (n_mfcc) = 40 coefficients
    mfc_coefficients=librosa.feature.mfcc(
        y=waveform, 
        sr=sample_rate, 
        n_mfcc=n_mfcc,
        n_fft=fft, 
        win_length=winlen, 
        window=window, 
        #hop_length=hop, 
        n_mels=mels, 
        fmax=sample_rate/2
        ) 

    return mfc_coefficients

def get_waveforms(file):
    
    # load an individual sample audio file
    # read the full 3 seconds of the file, cut off the first 0.5s of silence; native sample rate = 48k
    # don't need to store the sample rate that librosa.load returns
    waveform, _ = librosa.load(file, duration=3, offset=0.5, sr=sample_rate)
    
#    # Remove noise
#    waveform = librosa.effects.trim(waveform, top_db=20)[0]

#    # Remove silence
#    waveform, _ = librosa.effects.trim(waveform)
    

    # make sure waveform vectors are homogenous by defining explicitly
    waveform_homo = np.zeros((int(sample_rate*3,)))
    waveform_homo[:len(waveform)] = waveform
    
    # return a single file's waveform                                      
    return waveform_homo



def preprocess_audio(file, sample_rate=sample_rate):
        
    # get waveform
    waveform = get_waveforms(file)    
    
    # Reshape MFCCs into (num_frames, num_mfccs, 1) tensor
    # Extract MFCC features
    mfcc = feature_mfcc(waveform, sample_rate) 
    
    # need to make dummy input channel for CNN input feature tensor
    X = np.expand_dims(mfcc, 1)
    X = np.reshape(X, (1, 40, 282, 1))
    
#    # store shape so we can transform it back 
#    i,j,k,w = X.shape
   
    # Scaling the audio file
#    scaler = StandardScaler()
#    X = X.reshape(i, j*k) 
#    X = scaler.fit_transform(X)
    
    # Transform back to NxCxHxW 4D tensor format
#    X = X.reshape(i,j,k,w)

    return X

# Define a function to make prediction
def predict_emotion(file_path):
    
    # Preprocess audio
    X = preprocess_audio(file_path)

    # Load pre-trained model
    model = tf.keras.models.load_model('model.h5')

    # Make prediction
    emotion_prediction = model.predict(X)

    # Print predicted emotion
    emotion_labels = ['Surprise', 'Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust']
    predicted_emotion = emotion_labels[np.argmax(emotion_prediction)]
        
    return predicted_emotion

# Define Streamlit app
def app():
    st.title('Speech Emotion Recognition App')
    
    # Load an image from a file
    image = open("istockphoto-1318764563-612x612.jpg", "rb").read()
    
    # Display the image
    st.image(image, caption="Emotions", width=600)
    
    st.write('Click the "Record" button to start recording an audio sample')
    st.write('Click the "Predict" button to predict the emotion from the recorded audio sample')

    # Define variable to store audio file path
    file_path = ''

    # Define function to record audio
    def record():
        
        global file_path
        duration = 3  # recording duration in seconds
        sample_rate = 48000  # sample rate in Hz
        channels = 1  # number of audio channels

        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels)
        sd.wait()  # wait for the recording to finish

        file_path = "output.wav"
        sf.write(file_path, recording, sample_rate)

        return file_path

                
    # Define button for recording audio
    if st.button('Record'):
        file_path = record()
        st.write('Audio recording saved to', file_path)

    # Define button for predicting emotion
    if st.button('Predict'):
        file_path = record()
        if file_path:
            emotion = predict_emotion(file_path)
            if emotion == 'Surprise':
                st.write("<h1 style='color: orange; font-family: Arial'> Predicted emotion: Surprise! 😲 </h1>", unsafe_allow_html=True)        
            elif emotion == 'Neutral':
                st.write("<h1 style='color: grey; font-family: Arial'> Predicted emotion: Neutral 😐 </h1>", unsafe_allow_html=True)
            elif emotion == 'Calm':
                st.write("<h1 style='color: light blue; font-family: Arial'> Predicted emotion: Calm 😇 </h1>", unsafe_allow_html=True)
            elif emotion == 'Happy':
                st.write("<h1 style='color: yellow; font-family: Arial'> Predicted emotion: Happy 🤩 </h1>", unsafe_allow_html=True)
            elif emotion == 'Sad':
                st.write("<h1 style='color: purple; font-family: Arial'> Predicted emotion: Sad 😿 </h1>", unsafe_allow_html=True)
            elif emotion == 'Angry':
                st.write("<h1 style='color: red; font-family: Arial'> Predicted emotion: Angry 😤 </h1>", unsafe_allow_html=True)
            elif emotion == 'Fearful':
                st.write("<h1 style='color: dark blue; font-family: Arial'> Predicted emotion: Fearful 😨 </h1>", unsafe_allow_html=True)
            elif emotion == 'Disgust':
                st.write("<h1 style='color: brown; font-family: Arial'> Predicted emotion: Disgust 🤢 </h1>", unsafe_allow_html=True)
        else:
            st.write('No audio file found. Please record audio first.')

    # Load an image from a file
    image_1 = open("Asset-3pale.webp", "rb").read()
    
    # Display the image
    st.image(image_1, caption="SER", width=500)

# Run Streamlit app
if __name__ == '__main__':
    app()
    
#print(predict_emotion("output.wav"))
