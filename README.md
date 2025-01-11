# What is jinsei?
### !jinsei ? None : None # Always None
### Dataset

# Music Genre Classification & Guitar Chord Classification
### Description
This repository focuses on music genre classification and guitar chord classification using machine learning models. The project utilizes two key datasets:

1. **GTZAN Dataset for Music Genre Classification**:  
   This dataset contains 1,000 audio tracks, each 30 seconds long, covering 10 different music genres. It is used for training models to classify music genres. The dataset can be accessed on Kaggle at the following link:  
   [GTZAN Dataset - Music Genre Classification](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification).
2. **Guitar Chord Classification Dataset**:  
   This dataset contains labeled audio samples for several guitar chords. It is used for training a convolutional neural network (CNN) to classify guitar chords. The dataset can be accessed on Kaggle at:  
   [Guitar Chord Classification Using CNN](https://www.kaggle.com/datasets/fabianavinci/guitar-chords-v3).
3. **Music and Emotion (Sad and Happy) Dataset**:  
   This dataset is used for training models to classify musical emotions into categories like "sad" and "happy". Access it here:  
   [Musical Emotions Classification](https://www.kaggle.com/datasets/kingofarmy/musical-emotions-classification).
   
### Project Overview
- **Music Genre Classification**: The goal of this section is to classify music tracks into one of 10 genres using the GTZAN dataset. Various feature extraction techniques like **Mel-Frequency Cepstral Coefficients (MFCC)**, **Short-Time Fourier Transform (STFT)**, and **Log Spectrograms** are applied to the audio data. The project involves training models such as **CNNs** to process these features and predict the genre.
- **Guitar Chord Classification**: The objective here is to build a model to classify guitar chords using a CNN. The dataset provides different guitar chord sounds, and the model is trained to identify and classify the chord being played in a given audio clip.
- **Visualization and Preprocessing**:  
   For a deeper understanding of the data preprocessing steps, refer to and manipulate the `.ipynb` files.

### Notes
#### Audio Processing and Machine Learning / Deep Learning
1. **Waveform** represents audio data in the time domain, capturing amplitude variations over time.
2. **Feature Extraction Pipeline**:
   - **Raw Waveform to Frequency Domain**: 
     - The waveform’s power spectrum is processed using **FFT (Fast Fourier Transform)** to convert time-domain data into frequency information.
     - A **hop length** defines the step size for windowing, producing overlapping frames of FFT data.
   - **Output Features**:
     - **Spectrogram**: A time-frequency representation with frequency bins showing how energy varies across time and frequency.
     - **MFCC (Mel-frequency cepstral coefficients)**: Derived from the spectrogram using the Mel scale to mimic human auditory perception.
3. **Model Preferences**:
   - **Spectrogram** data is best suited for **Convolutional Neural Networks (CNNs)** because it is a 2D image-like representation. CNNs can efficiently learn spatial patterns and hierarchies in time-frequency data.
   - **MFCCs** are best suited for **Recurrent Neural Networks (RNNs) and LSTM models**, which are designed to capture sequential dependencies in time-series data, making them ideal for modeling temporal variations in audio signals.
4. This pipeline transforms raw audio data into structured inputs for deep learning models, enabling tasks such as genre classification, chord recognition, and emotion detection.

#### TODO
1. **Implement "Music and Emotion (Sad and Happy)" Classification**
   - Consider using **spectrograms with CNN + LSTM** to capture both spatial frequency patterns and temporal dependencies for richer, detailed emotion modeling.
   - Alternatively, explore **MFCCs with CNN + LSTM** for a more compact, human-centric feature set that emphasizes pitch and timbre.
   - Evaluate performance trade-offs between these approaches and document insights for future reference.

2. **Organize and Review Learnings**
   - Compile detailed notes on preprocessing, feature extraction, and model architectures.
   - Summarize key findings from experiments and observations to refine future models.

3. **Deepen Understanding of Audio-Visual Integration**
   - Investigate potential applications combining **music and image data** to enhance multi-modal emotion recognition systems.
   - Research additional **deep learning architectures** suited for audio-visual data fusion to expand model capabilities.