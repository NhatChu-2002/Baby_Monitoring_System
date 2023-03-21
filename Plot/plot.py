import librosa
import matplotlib.pyplot as plt
import os
noise_mfccs = []
crying_mfccs = []
laugh_mfccs = []
silence_mfccs = []

# loop through each sound category folder
for category in ['D:\\HK2-Năm 3\\PBL5\\Code\\data\\Crying',
                'D:\\HK2-Năm 3\\PBL5\\Code\\data\\Laugh',
                'D:\\HK2-Năm 3\\PBL5\\Code\\data\\Noise',
                'D:\\HK2-Năm 3\\PBL5\\Code\\data\\Silence']:
    # loop through each sound file in the folder
    for sound_file in os.listdir(category):
        # load the sound file and extract mfcc features
        y, sr = librosa.load(os.path.join(category, sound_file))
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # add mfcc features to corresponding array based on sound category
        if category == 'D:\\HK2-Năm 3\\PBL5\\Code\\data\\Noise':
            noise_mfccs.append(mfccs)
        elif category == 'D:\\HK2-Năm 3\\PBL5\\Code\\data\\Crying':
            crying_mfccs.append(mfccs)
        elif category == 'D:\\HK2-Năm 3\\PBL5\\Code\\data\\Laugh':
            laugh_mfccs.append(mfccs)
        else:
            silence_mfccs.append(mfccs)
            
# create subplots for each sound category
fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(10, 10))

# set titles for each subplot
axs[0].set_title('Noise')
axs[1].set_title('Crying')
axs[2].set_title('Laugh')
axs[3].set_title('Silence')

# plot mfcc features for each sound category
im0 = axs[0].imshow(librosa.power_to_db(noise_mfccs[1]), cmap='coolwarm')
im1 = axs[1].imshow(librosa.power_to_db(crying_mfccs[1]), cmap='coolwarm')
im2 = axs[2].imshow(librosa.power_to_db(laugh_mfccs[1]), cmap='coolwarm')
im3 = axs[3].imshow(librosa.power_to_db(silence_mfccs[1]), cmap='coolwarm')

# add colorbars to subplots
fig.colorbar(im0, ax=axs[0])
fig.colorbar(im1, ax=axs[1])
fig.colorbar(im2, ax=axs[2])
fig.colorbar(im3, ax=axs[3])

plt.show()



