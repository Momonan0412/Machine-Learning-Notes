import os
import librosa
import math
import json
class AudioPreprocessor:
    """
    dataset_path: str
        The file path to the dataset containing the audio files.
        This is where the raw audio data will be loaded from.

    json_path: str
        The file path where the processed data will be saved in JSON format.
        This is typically used to store features extracted from the audio.

    sample_rate: int, optional, default=22050
        The sampling rate used to load the audio files. This determines the number of samples per second of audio.
        If the provided audio files have a different sample rate, they will be resampled to this rate.
        For more details, refer to the `librosa.load` documentation: https://librosa.org/doc/main/generated/librosa.load.html

    num_segments: int, optional, default=5
        The number of segments to divide each audio file into for processing.
        This is a form of data augmentation that helps increase the diversity of the training set while avoiding overfitting.
        Smaller segments preserve localized features in the audio, making the extracted data more meaningful and enhancing generalization.

    n_mfcc: int, optional, default=13
        The number of Mel-frequency cepstral coefficients (MFCCs) to extract from each audio segment.
        MFCCs are a popular feature used to represent the timbre of audio, capturing the frequency characteristics of sound.

    n_fft: int, optional, default=2048
        The window size (in samples) for performing the Fast Fourier Transform (FFT).
        This determines the number of data points used for each FFT analysis, affecting the frequency resolution of the resulting spectrogram.

    hop_length: int, optional, default=512
        The step size (in samples) between successive frames when performing the FFT.
        This controls the overlap between windows and influences the time resolution of the spectrogram.
    """
    def __init__(self, dataset_path, json_path, 
                 sample_rate=22050, num_segments=10, n_mfcc=12, n_fft=2048, hop_length=512):
        self._dataset_path = dataset_path
        self._json_path = json_path
        self._sample_rate = sample_rate
        self._num_segments = num_segments
        self._n_mfcc = n_mfcc
        self._n_fft = n_fft
        self._hop_length = hop_length
        self._dict_data_storage()
        self._set_duration()
        self._set_sample_per_track()
        self._set_num_samples_per_segment()
        self._set_expected_num_mfcc_vectors_per_segment()
        
    def _save_mfcc(self):
        # Loop Through All Genres
        
        # print(self._duration)
        # print(self._sample_per_track)
        # print(self._num_samples_per_segment)
        # print(self._expected_num_mfcc_vectors_per_segment)
        
        for i, (dirpath, dirnames, filenames) in enumerate(os.walk(self._dataset_path)):
            # print("LABEL!", i)
            if dirpath != self._dataset_path: # os.walk gives "self._dataset_path" for the first iteration
                # print("PATH!", dirpath)
                self._save_semantic_label(dirpath)
                self._process_file_genre(filenames, dirpath, i-2)
        
        print("Finish!")
    
    def _process_file_genre(self, filenames, dirpath, label):
        for filename in filenames:
            # Access only the audio file
            if ".wav" in filename:
                file_path = os.path.join(dirpath, filename)
                # print(file_path) # Debugging
                if "jazz.00054" in file_path:
                    # print("SHOULD BE! jazz.00054!", file_path)
                    continue
                signal, sample_rate = librosa.load(file_path, sr=self._sample_rate)
                # """Debugging"""
                # print(signal)
                # print(sample_rate)
                # print(label)
                # print("---")
                self._process_segment(signal=signal, label=label)
    
    """Given that all dataset's duration is 30, hence the default value"""
    def _set_duration(self, duration=30):
        self._duration = duration
    
    """
    22050 * 30
    Visualize this as the number of "index" in the signal matrix
    """
    def _set_sample_per_track(self):
        self._sample_per_track = self._sample_rate * self._duration

    """
    Divide the `number of "index" in the signal matrix` to segments using the self._num_segments
    """
    def _set_num_samples_per_segment(self):
        self._num_samples_per_segment = int(self._sample_per_track / self._num_segments)
    
    """
    Using the self._num_samples_per_segment / self._hop_length we can get the approximate 
    expected length of the mfcc vector
    """
    def _set_expected_num_mfcc_vectors_per_segment(self):
        self._expected_num_mfcc_vectors_per_segment = math.ceil(self._num_samples_per_segment / self._hop_length)

    def _process_segment(self, signal, label):
        for segment in range(self._num_segments):
            start_sample = self._num_samples_per_segment * segment
            # segment=0 => 0
            # segment=1 => self._num_samples_per_segment * 1
            finish_sample = start_sample + self._num_samples_per_segment
            # segment=0 => start_sample + self._num_samples_per_segment
            # Therefore, start_sample and finish_sample is the "segment-step" in traversing the "signal" array
            mfcc = librosa.feature.mfcc(y=signal[start_sample:finish_sample], 
                                        sr=self._sample_rate, # Should match the rate used to load the signal
                                        n_fft=self._n_fft, 
                                        n_mfcc=self._n_mfcc, 
                                        hop_length=self._hop_length)
            mfcc = mfcc.T # For Easy Manipulation
            if len(mfcc) == self._expected_num_mfcc_vectors_per_segment:
                self._data["mfcc"].append(mfcc.tolist())
                self._data["label"].append(label)
        # print("Label #", label)
        # print("Finish!")
    
    def _save_semantic_label(self, dirpath):
        if "genres" in dirpath:
            # print(dirpath)
            dirpath_components = dirpath.split("\\")
            if len(dirpath_components) == 4: # Valid "dirpath_components" are with length 4
                semantic_label = dirpath_components[-1]
                self._data["mapping"].append(semantic_label)
                print("Processing", semantic_label) # Debugging
    
    """Dictionary to store data"""
    def _dict_data_storage(self):
        self._data = {
            "mapping" : [], #
            "mfcc" : [], # Training Inputs 
            "label" : [] # Outputs or Targets
        }
        
    def _save_us_json_file(self):
        with open(self._json_path, "w") as data_as_json_file:
            json.dump(self._data, data_as_json_file, indent=4)
        
if __name__ == "__main__":
    DATASET_PATH = "Music Genre Classification\\Data"
    JSON_PATH = "Music Genre Classification\\data.json"
    audio_preprocessor = AudioPreprocessor(DATASET_PATH, JSON_PATH)
    audio_preprocessor._save_mfcc()
    audio_preprocessor._save_us_json_file()
    
    # Open and read the JSON file
    # with open(JSON_PATH, "r") as data_file:
    #     data = json.load(data_file)

    # print(f"The JSON file has {len(data["mapping"])} items.")
    # print(f"The JSON file has {len(data["mfcc"])} items.")
    # print(f"The JSON file has {len(data["label"])} items.")