import json
import numpy as np
from sklearn.model_selection import train_test_split
from keras.api import Sequential
class DataUtils:
    def __init__(self, dataset_path):
        self._load_data(dataset_path)
        self._convert_list_to_array()
        self._convert_maps_to_array()
        
    def _load_data(self, dataset_path):
        with open(dataset_path, "r") as data_set_file_path:
            self._data = json.load(data_set_file_path)
            
    def _convert_list_to_array(self):
        self._inputs = np.array(self._data["mfcc"])
        self._targets = np.array(self._data["label"])
    
    def _convert_maps_to_array(self):
        self._map = np.array(self._data["mapping"])
        
    def _get_inputs_and_targets(self):
        return self._inputs, self._targets
    
    def _get_maps(self):
        return self._map
    
    def prepare_data(self, test_size, validation_size):
        # Load Data
        x, y = self._get_inputs_and_targets()
        # Split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
        # Train Validation Split
        x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=validation_size)
        
        # CNN Need 3D Array If Audio ("Samples/Hoplength","MFCC", "Depth/Channel")
        # Current "x_train", "x_validation" and "x_test"'s Shape (5991, 130, 12) ("Total Sample", "Samples/Hoplength", "MFCC")
        # It Does Not Contain "Depth/Channel"
        # Apply https://stackoverflow.com/questions/29241056/how-do-i-use-np-newaxis
        x_train = x_train[..., np.newaxis]
        x_validation = x_validation[..., np.newaxis]
        x_test = x_test[..., np.newaxis]
        print("X Train's Shape ", x_train.shape)
        print("X Validation's Shape ", x_validation.shape)
        print("X Test's Shape ", x_test.shape)
        print("X Train's Shape ", y_train.shape)
        print("X Validation's Shape ", y_validation.shape)
        print("X Test's Shape ", y_test.shape)
        return x_train, x_validation, x_test, y_train, y_validation, y_test
    
    def data_predict(self, model, input):
        # inputs's current dimension => [Datas, N_MFCCs, Depth]
        # Needed For CNN prediction => [1, Datas, N_MFCCs, Depth]
        # Hence
        input = input[np.newaxis, ...]
        if isinstance(model, Sequential):
            prediction = model.predict(input)
            # Extract index with the max value, since the activation function is "softmax"
            # predicted_index = np.argmax(prediction, axis=1)
            predicted_index = np.argmax(prediction, axis=1)
            return predicted_index
        else:
            raise TypeError("The model is not a Sequential model.")

        
        