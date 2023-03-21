import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
# Đường dẫn đến các thư mục chứa các file âm thanh
noise_dir = "D:\\HK2-Năm 3\\PBL5\\Code\\data\\Noise"
crying_dir = "D:\\HK2-Năm 3\\PBL5\\Code\\data\\Crying"
laugh_dir = "D:\\HK2-Năm 3\\PBL5\\Code\\data\\laugh"
silence_dir = "D:\\HK2-Năm 3\\PBL5\\Code\\data\\silence"

# Tổng số hệ số MFCC muốn trích xuất
n_mfcc = 13

# Danh sách các đặc trưng MFCC cho các thư mục
noise_mfcc = []
crying_mfcc = []
silence_mfcc = []
laugh_mfcc = []

# Lặp lại qua các file âm thanh trong thư mục và tính toán đặc trưng MFCC
for folder, mfcc_list in zip([noise_dir, crying_dir, silence_dir, laugh_dir], [noise_mfcc, crying_mfcc, silence_mfcc, laugh_mfcc]):
    for file in os.listdir(folder):
        # Tải tệp âm thanh và tính toán đặc trưng MFCC
        y, sr = librosa.load(os.path.join(folder, file), sr=44100, dtype='float32')
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_list.append(mfcc.T) # Transpose the mfcc matrix to get the shape (n_frames, n_mfcc)

# Convert the lists of mfcc arrays into numpy arrays
noise_mfcc = np.vstack(noise_mfcc)
crying_mfcc = np.vstack(crying_mfcc)
silence_mfcc = np.vstack(silence_mfcc)
laugh_mfcc = np.vstack(laugh_mfcc)


# Danh sách các mảng MFCC và nhãn tương ứng
mfcc_list = [noise_mfcc.ravel(), crying_mfcc.ravel(), silence_mfcc.ravel(), laugh_mfcc.ravel()]
labels = ['Noise', 'Crying', 'Silence', 'Laugh']

# Vẽ boxplot
fig, ax = plt.subplots()
ax.boxplot(mfcc_list, labels=labels)
ax.set_title('MFCC Comparison')
ax.set_xlabel('Category')
ax.set_ylabel('MFCC Value')
plt.show()

# # Vẽ violin plot
# fig, ax = plt.subplots()
# ax.violinplot(mfcc_list, showmeans=True)
# ax.set_title('MFCC Comparison')
# ax.set_xlabel('Category')
# ax.set_ylabel('MFCC Value')
# ax.set_xticklabels(labels)
# plt.show()
