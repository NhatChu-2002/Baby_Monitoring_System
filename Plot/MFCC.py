import librosa
import numpy as np
import matplotlib.pyplot as plt
import os

def extract_mfcc(file_path):
    y, sr = librosa.load(file_path)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean

folder_names = ['D:\\HK2-Năm 3\\PBL5\\Code\\data\\Crying',
                'D:\\HK2-Năm 3\\PBL5\\Code\\data\\Laugh',
                'D:\\HK2-Năm 3\\PBL5\\Code\\data\\Noise',
                'D:\\HK2-Năm 3\\PBL5\\Code\\data\\Silence']
mfccs = []
labels = []

for i, folder in enumerate(folder_names):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        mfcc = extract_mfcc(file_path)
        mfccs.append(mfcc)
        labels.append(i)
        
plt.figure(figsize=(12, 6))
plt.title("MFCC Comparison")
plt.xlabel("MFCC Coefficient")
plt.ylabel("Mean Value")
plt.ylim(-20, 20)

colors = ['blue', 'green', 'orange', 'red']

for i in range(len(folder_names)):
    mfccs_filtered = [mfccs[j] for j in range(len(mfccs)) if labels[j] == i]
    mfccs_mean = np.mean(mfccs_filtered, axis=0)
    plt.plot(mfccs_mean, color=colors[i], label=folder_names[i])
    
plt.legend()
plt.show()