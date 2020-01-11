import pandas as pd
import os

root_data_dir = 'activity_recognition'

# Assigning labels to each flower category
num_classes = 2
labels_name={'class_11':0,'class_19':1}

# Loop over every activity category in train and test
train_data_path = os.path.join('activity_data','train')
test_data_path = os.path.join('activity_data','test')

if not os.path.exists('data_files'):
    os.mkdir('data_files')
if not os.path.exists('data_files/train'):
    os.mkdir('data_files/train')
if not os.path.exists('data_files/test'):
    os.mkdir('data_files/test')

data_dir_list = os.listdir(train_data_path)
for data_dir in data_dir_list:
    label = labels_name[str(data_dir)]
    video_list = os.listdir(os.path.join(train_data_path,data_dir))
    for vid in video_list:
        train_df = pd.DataFrame(columns=['FileName', 'Label', 'ClassName'])
        img_list = os.listdir(os.path.join(train_data_path,data_dir,vid))
        for img in img_list:
            img_path = os.path.join(train_data_path,data_dir,vid,img)
            train_df = train_df.append({'FileName': img_path, 'Label': label,'ClassName':data_dir },ignore_index=True)
        file_name='{}_{}.csv'.format(data_dir,vid)
        train_df.to_csv('data_files/train/{}'.format(file_name))

data_dir_list = os.listdir(test_data_path)
for data_dir in data_dir_list:
    label = labels_name[str(data_dir)]
    video_list = os.listdir(os.path.join(test_data_path,data_dir))
    for vid in video_list:
        test_df = pd.DataFrame(columns=['FileName', 'Label', 'ClassName'])
        img_list = os.listdir(os.path.join(test_data_path,data_dir,vid))
        for img in img_list:
            img_path = os.path.join(test_data_path,data_dir,vid,img)
            test_df = test_df.append({'FileName': img_path, 'Label': label,'ClassName':data_dir },ignore_index=True)
        file_name='{}_{}.csv'.format(data_dir,vid)
        test_df.to_csv('data_files/test/{}'.format(file_name))

