import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


model = load_model('models/emotion_speech_recognition_model.h5')
EMOTIONS = ['angry', 'happy', 'neutral', 'sad']

st.title("TESS Toronto Emotional Speech Dataset with Emotion Recognition")
st.write("Explore audio files and recognize emotions using a pre-trained model.")


uploaded_file = st.file_uploader("Upload a Speech Audio File", type=["wav"])

if uploaded_file is not None:
    st.write(f"Uploaded File: {uploaded_file.name}")
    st.audio(uploaded_file, format='audio/wav')
    audio_data, sr = librosa.load(uploaded_file)
    
    st.write("Waveform of the Uploaded Audio File:")
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(audio_data, sr=sr)
    plt.title('Waveform of Audio')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    st.pyplot(plt)

    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40)
    mfccs_processed = np.mean(mfccs.T, axis=0) 

    mfccs_processed = np.expand_dims(mfccs_processed, axis=0)
    mfccs_processed = np.expand_dims(mfccs_processed, axis=-1)

    predictions = model.predict(mfccs_processed)
    predicted_emotion = EMOTIONS[np.argmax(predictions)]

    st.write(f"Predicted Emotion: {predicted_emotion}")
