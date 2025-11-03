import pickle
import os

DATA_DIR = './data'

# Mở file pickle
with open('./data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

data = data_dict['data']
labels = data_dict['labels']

# Duyệt tất cả vector
for i, vec in enumerate(data):
    if len(vec) != 42:  # 21 landmark * 2 (x, y)
        # Tìm ảnh tương ứng trong thư mục
        label = labels[i]
        folder = os.path.join(DATA_DIR, label)
        img_files = os.listdir(folder)
        print(f"Ảnh có vector sai kích thước: {img_files[i % len(img_files)]} (nhãn: {label}, length: {len(vec)})")
