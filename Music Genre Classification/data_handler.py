import json
import numpy as np
class DataUtils:
    def __init__(self, dataset_path):
        self._load_data(dataset_path)
        self._convert_list_to_array()
        
    def _load_data(self, dataset_path):
        with open(dataset_path, "r") as data_set_file_path:
            self._data = json.load(data_set_file_path)
            
    def _convert_list_to_array(self):
        self._inputs = np.array(self._data["mfcc"])
        self._targets = np.array(self._data["label"])
        
    def _get_inputs_and_targets(self):
        return self._inputs, self._targets