import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

noise_dir = "D:\\HK2-Năm 3\\PBL5\\Code\\data\\Noise"
crying_dir = "D:\\HK2-Năm 3\\PBL5\\Code\\data\\Crying"
laugh_dir = "D:\\HK2-Năm 3\\PBL5\\Code\\data\\laugh"
silence_dir = "D:\\HK2-Năm 3\\PBL5\\Code\\data\\silence"

def extract_mfcc(file_path):
    y, sr = librosa.load(file_path)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12)
    return mfccs

def extract_features(directory):
    features = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        mfccs = extract_mfcc(file_path)
        features.append(mfccs)
    return features

noise_features = extract_features(noise_dir)
crying_features = extract_features(crying_dir)
silence_features = extract_features(silence_dir)
laugh_features = extract_features(laugh_dir)

noise_features = np.concatenate(noise_features, axis=1)
crying_features = np.concatenate(crying_features, axis=1)
silence_features = np.concatenate(silence_features, axis=1)
laugh_features = np.concatenate(laugh_features, axis=1)
noise_mean = np.mean(noise_features, axis=1)
noise_std = np.std(noise_features, axis=1)
crying_mean = np.mean(crying_features, axis=1)
crying_std = np.std(crying_features, axis=1)
silence_mean = np.mean(silence_features, axis=1)
silence_std = np.std(silence_features, axis=1)
laugh_mean = np.mean(laugh_features, axis=1)
laugh_std = np.std(laugh_features, axis=1)

# Plot histograms for each feature
plt.figure(figsize=(10,10))
for i in range(12):
    plt.subplot(4, 5, i+1)
    plt.hist(noise_features[i], bins=20, alpha=0.5, color='red', label='Noise')
    plt.hist(crying_features[i], bins=20, alpha=0.5, color='blue', label='Crying')
    plt.hist(silence_features[i], bins=20, alpha=0.5, color='green', label='Silence')
    plt.hist(laugh_features[i], bins=20, alpha=0.5, color='purple', label='Laugh')
    plt.legend()
    plt.title("MFCC {}".format(i+1))
plt.tight_layout()
plt.show()
