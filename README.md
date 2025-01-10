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
### Project Overview
- **Music Genre Classification**: The goal of this section is to classify music tracks into one of 10 genres using the GTZAN dataset. Various feature extraction techniques like **Mel-Frequency Cepstral Coefficients (MFCC)**, **Short-Time Fourier Transform (STFT)**, and **Log Spectrograms** are applied to the audio data. The project involves training models such as **CNNs** to process these features and predict the genre.
- **Guitar Chord Classification**: The objective here is to build a model to classify guitar chords using a CNN. The dataset provides different guitar chord sounds, and the model is trained to identify and classify the chord being played in a given audio clip.
- **Visualization and Preprocessing**:  
   For a deeper understanding of the data preprocessing steps, refer to and manipulate the `.ipynb` files.

