import os
import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
import math
import json
import h5py
STATIC_PATH = "Guitar Chord Classification\\"
class ChordFeatureExtraction:
    def __init__(self, dataset_path, h5py_path,
                 n_fft = 2048, hop_length = 512, num_segments = 3, sample_rate = 22050):
        self._dataset_path = dataset_path
        self._h5py_path = h5py_path
        self._n_fft = n_fft
        self._hop_length = hop_length
        self._num_segments = num_segments
        self._sample_rate = sample_rate
        
        self._set_data()
        self._segments_handler()
        
    def _set_data(self):
        self._data = {
            "mapping" : [],
            "spectrogram" : [],
            "label" : []
        }
    
    def _segments_handler(self):
        self._duration = 5
        self._sample_per_track = self._sample_rate * self._duration
        self._num_samples_per_segment = int(self._sample_per_track / self._num_segments)
        self._expected_num_mfcc_vectors_per_segment = math.ceil(self._num_samples_per_segment / self._hop_length)
        
    def _process_spectrogram(self):    
        for i, (dirpath, dirnames, filenames) in enumerate(os.walk(self._dataset_path)):
            if dirpath != self._dataset_path:
                map = dirpath.split('\\')[-1]
                self._data['mapping'].append(map)
                for filename in filenames:
                    file_path = os.path.join(dirpath, filename)
                    signal, sample_rate = librosa.load(file_path, sr=22050)
                    for segment in range(self._num_segments):
                        start_sample = self._num_samples_per_segment * segment
                        finish_sample = start_sample + self._num_samples_per_segment
                        if len(signal[start_sample:finish_sample]) < self._n_fft:
                            # print("Empty ", signal[start_sample:finish_sample])
                            continue
                        stft = librosa.core.stft(signal[start_sample:finish_sample], hop_length=self._hop_length, n_fft=self._n_fft)
                        spectrogram = np.abs(stft)
                        log_spectrogram = librosa.amplitude_to_db(spectrogram)
                        log_spectrogram = log_spectrogram.T
                        if len(log_spectrogram) == self._expected_num_mfcc_vectors_per_segment:
                            target_shape = log_spectrogram.shape
                            # print(target_shape)
                        if len(log_spectrogram) < self._expected_num_mfcc_vectors_per_segment:
                            padded_log_spectrogram  = np.zeros(target_shape)
                            # print(log_spectrogram.shape[0])
                            
                            # places the original values into the padded array,
                            # keeping its original values intact
                            # while adding zeros to fill the rest.
                            
                            # [:log_spectrogram.shape[0], :] Given the indices replace is with the log_spectrogram
                            
                            padded_log_spectrogram[:log_spectrogram.shape[0], :] = log_spectrogram
                            log_spectrogram = padded_log_spectrogram
                            # print(padded_log_spectrogram.shape)
                        # print(type(log_spectrogram))
                        if len(log_spectrogram) == self._expected_num_mfcc_vectors_per_segment:
                            self._data['spectrogram'].append(log_spectrogram.tolist())
                            self._data['label'].append(i-1)
    def _get_data(self):
        return self._data
    
    # Open or create an HDF5 file
    def _data_to_h5py(self, dataset_type):
        with h5py.File(self._h5py_path, 'w') as h5_file:
            # Save each data component into a dataset
            h5_file.create_dataset('mapping', data=np.array(self._data['mapping'], dtype='S'))   # Storing strings
            h5_file.create_dataset('spectrogram', data=np.array(self._data['spectrogram']))      # Assuming 2D spectrogram arrays
            h5_file.create_dataset('label', data=np.array(self._data['label'], dtype=int))       # Labels as integers

            # Optionally, save metadata if needed
            h5_file.attrs['description'] = 'Guitar chord classification dataset ' + dataset_type
            h5_file.attrs['num_classes'] = len(self._data['mapping'])
            
if __name__ == "__main__":
    extraction = ChordFeatureExtraction(dataset_path = STATIC_PATH + "Dataset\\Test",
                                        h5py_path= STATIC_PATH + "test_chords_data.h5")
    extraction._process_spectrogram()
    extraction._data_to_h5py("- Test")
    print("Test Chords Data Shape", np.array(extraction._get_data()['spectrogram']).shape)
    
    extraction = ChordFeatureExtraction(dataset_path = STATIC_PATH + "Dataset\\Training",
                                        h5py_path= STATIC_PATH + "train_chords_data.h5")
    extraction._process_spectrogram()
    extraction._data_to_h5py("- Train")
    print("Train Chords Data Shape", np.array(extraction._get_data()['spectrogram']).shape)
    