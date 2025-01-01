import os
import librosa
class AudioPreprocessor:
    """
    dataset_path: path to dataset
    json_path: storage
    sample_rate: 
        # https://librosa.org/doc/main/generated/librosa.load.html
        # Audio will be automatically resampled to the given rate (default sr=22050).
        # Explicit Parameter => sr=22050
    num_segments: data augmentation
    """
    def __init__(self, dataset_path, json_path, sample_rate=22050, num_segments=5):
        self._dataset_path = dataset_path
        self._json_path = json_path
        self._sample_rate = sample_rate
        self._num_segments = num_segments
        self._dict_data_storage()
    """
    n_mfcc: # of coefficients
    n_fft: window size when fft is performed
    hop_length: the step of the window in performing fft
    """
    def _save_mfcc(self, n_mfcc=13, n_fft=2048, hop_length=512):
        # Loop Through All Genres
        for i, (dirpath, dirnames, filenames) in enumerate(os.walk(self._dataset_path)):
            if dirpath != self._dataset_path: # os.walk gives "self._dataset_path" for the first iteration
                # self._save_semantic_label(dirpath)
                self._process_file_genre(filenames, dirpath)
                pass
    
    def _process_file_genre(self, filenames, dirpath):
        for filename in filenames:
            # Access only the audio file
            if ".wav" in filename:
                file_path = os.path.join(dirpath, filename)
                # print(file_path) # Debugging
                if "jazz.00054" in file_path:
                    print("SHOULD BE! jazz.00054!", file_path)
                    continue
                signal, sample_rate = librosa.load(file_path, sr=self._sample_rate)
                """Debugging"""
                # print(signal)
                # print(sample_rate)
                # print("---")
    
    def _process_segment(self):
        for segment in range(self._num_segments):
            pass
        pass
    
    def _extracting_mfcc(self):
        pass
    
    
    def _save_semantic_label(self, dirpath):
        dirpath_components = dirpath.split("\\")
        if(len(dirpath_components) == 4): # Valid "dirpath_components" are with length 4
            semantic_label = dirpath_components[-1]
            self._data["mapping"].append(semantic_label)
            print(semantic_label) # Debugging
    
    """Dictionary to store data"""
    def _dict_data_storage(self):
        self._data = {
            "mapping" : [], #
            "mfcc" : [], # Training Inputs 
            "label" : [] # Outputs or Targets
        }
        
if __name__ == "__main__":
    DATASET_PATH = "Music Genre Classification\\Data"
    JSON_PATH = "data.json"
    audio_preprocessor = AudioPreprocessor(DATASET_PATH, JSON_PATH)
    audio_preprocessor._save_mfcc()