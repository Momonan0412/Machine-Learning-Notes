import sys
import os
import librosa
import numpy as np
import math
import h5py
if __name__ == "__main__":
    test = "Test"
    train = "Train"
    # dataset_path = "Music and Emotion (Sad and Happy)\\Dataset\\Audio_Files\\" + test
    dataset_path = "Music and Emotion (Sad and Happy)\\Dataset\\Audio_Files\\" + train
    data = {
        "mapping" : [],
        "mfcc" : [],
        "label" : []
    }
    hop_length = 512
    num_segments = 3
    duration = 10
    sample_rate = 22050
    sample_per_track = duration * sample_rate
    num_samples_per_segment = sample_per_track // num_segments
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        if dataset_path != dirpath:
            # print(dirnames) # Empty
            path_to_append = dirpath
            dirpath = dirpath.split('\\')
            print("Mapping:", dirpath[-1]) # mapping
            data['mapping'].append(dirpath[-1])
            # print("Length:", len(dirpath[-1])) # length
            for filename in filenames:
                file_path = os.path.join(path_to_append, filename)
                # print("File Path: ", file_path)
                try:
                    signal, sample_rate = librosa.load(file_path, sr=22050)
                    # if(len(signal) < 2048):
                    #     break
                    # duration_checker = librosa.get_duration(y=signal, sr=sample_rate)
                    # print(duration)
                    # if(duration_checker < 10):
                    #     print(duration_checker)
                    #     break
                    
                    for segment in range(num_segments):
                        start_signal = num_samples_per_segment * segment
                        finish_signal = segment + num_samples_per_segment
                        mfcc = librosa.feature.mfcc(y=signal[start_signal:finish_signal], sr=sample_rate, n_mfcc=13, n_fft=2048, hop_length=hop_length)
                        mfcc = mfcc.T
                        # print(len(mfcc), " and ", expected_num_mfcc_vectors_per_segment)
                        if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                            mfcc_shape = mfcc.shape
                        if len(mfcc) < expected_num_mfcc_vectors_per_segment:
                            padd_mfcc = np.zeros(mfcc_shape)
                            padd_mfcc[:mfcc.shape[0], :] = mfcc
                            mfcc = padd_mfcc
                        # print("Data! ", mfcc.shape)
                        data['mfcc'].append(mfcc)
                        # print("Label! ", i - 1)
                        data['label'].append(mfcc.tolist())
                except Exception as e:
                    print(f"Skipping {file_path}: Error loading file - {e}")
                    continue
    
    print(np.array(data["mfcc"]).shape)
    print(np.array(data["label"]).shape)
    # h5py_path = "Music and Emotion (Sad and Happy)\\"+ test +"_sad_and_happy_dataset.h5"
    h5py_path = "Music and Emotion (Sad and Happy)\\"+ train +"_sad_and_happy_dataset.h5"
    with h5py.File(h5py_path, 'w') as h5_file:
        h5_file.create_dataset('mapping', data=np.array(data['mapping'], dtype='S'))
        h5_file.create_dataset('mfcc', data=np.array(data['mfcc']))
        h5_file.create_dataset('label', data=np.array(data['label'], dtype=int))