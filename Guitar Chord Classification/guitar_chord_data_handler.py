import json
import numpy as np
from sklearn.model_selection import train_test_split
from keras.api import Sequential
import h5py
class GuitarChordDataHandler:
    def __init__(self, h5py_path):
        self._h5py_path = h5py_path
        self._load_data(self._h5py_path)
        
    def _load_data(self, dataset_path):
        with h5py.File(self._h5py_path, 'r') as h5_file:
            self._inputs = h5_file["spectrogram"][:]
            self._targets = h5_file["label"][:]
            # self._mapping = [x.decode('utf-8') for x in h5_file["mapping"][:]]

            self._map = np.array([])
            for map in h5_file["mapping"][:]:
                # print("Map: ", map.decode('utf-8') , " and Type: ", type(map.decode('utf-8')))
                decoded_map = map.decode('utf-8')
                self._map = np.append(self._map, decoded_map)
        
    def _get_inputs_and_targets(self):
        return self._inputs, self._targets
    
    def _get_maps(self):
        return self._map
    