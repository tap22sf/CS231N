import numpy as np
import pandas as pd
import os
import random 
from shutil import copyfile
import pydicom as dicom
import cv2

COPY_FILE = True

seed = 0
np.random.seed(seed) # Reset the seed so all runs are the same.
random.seed(seed)
MAXVAL = 255  # Range [0 255]

# Path to the combined dataset (train, val and test)
root = 'G:/combinedDataset'
savepath = root + '/data'

if (COPY_FILE):
    savepath = root + '/data'
    if(not os.path.exists(savepath)):
        os.makedirs(savepath)
    savepath = root + '/data/train'
    if(not os.path.exists(savepath)):
        os.makedirs(savepath)
    savepath = root + '/data/test'
    if(not os.path.exists(savepath)):
        os.makedirs(savepath)

# Paths to the downloaded datasets
covidPath = 'G:/DownloadedDatasets/covid-chestxray-dataset'
kaggle_datapath = 'G:/DownloadedDatasets/rsna-pneumonia-detection-challenge'

# path to covid-19 dataset from https://github.com/ieee8023/covid-chestxray-dataset
imgpath = covidPath + '/images' 
csvpath = covidPath + '/metadata.csv'

# path to https://www.kaggle.com/c/rsna-pneumonia-detection-challenge
kaggle_csvname = 'stage_2_detailed_class_info.csv' # get all the normal from here
kaggle_csvname2 = 'stage_2_train_labels.csv' # get all the 1s from here since 1 indicate pneumonia
kaggle_imgpath = 'stage_2_train_images'

# parameters for COVIDx dataset
train = []
test = []
test_count = {'normal': 0, 'pneumonia': 0, 'COVID-19': 0}
train_count = {'normal': 0, 'pneumonia': 0, 'COVID-19': 0}

mapping = dict()
mapping['COVID-19'] = 'COVID-19'
mapping['SARS'] = 'pneumonia'
mapping['MERS'] = 'pneumonia'
mapping['Streptococcus'] = 'pneumonia'
mapping['Normal'] = 'normal'
mapping['Lung Opacity'] = 'pneumonia'
mapping['1'] = 'pneumonia'

# train/test split
split = 0.1

# adapted from https://github.com/mlmed/torchxrayvision/blob/master/torchxrayvision/datasets.py#L814
csv = pd.read_csv(csvpath, nrows=None)
idx_pa = csv["view"] == "PA"  # Keep only the PA view
csv = csv[idx_pa]

pneumonias = ["COVID-19", "SARS", "MERS", "ARDS", "Streptococcus"]
pathologies = ["Pneumonia","Viral Pneumonia", "Bacterial Pneumonia", "No Finding"] + pneumonias
pathologies = sorted(pathologies)

# get non-COVID19 viral, bacteria, and COVID-19 infections from covid-chestxray-dataset
# stored as patient id, image filename and label
filename_label = {'normal': [], 'pneumonia': [], 'COVID-19': []}
count = {'normal': 0, 'pneumonia': 0, 'COVID-19': 0}
print(csv.keys())
for index, row in csv.iterrows():
    f = row['finding']
    if f in mapping:
        count[mapping[f]] += 1
        entry = [int(row['patientid']), row['filename'], mapping[f]]
        filename_label[mapping[f]].append(entry)

print('Data distribution from covid-chestxray-dataset:')
print(count)

# %%
# add covid-chestxray-dataset into COVIDx dataset
# since covid-chestxray-dataset doesn't have test dataset
# split into train/test by patientid
# for COVIDx:
# patient 8 is used as non-COVID19 viral test
# patient 31 is used as bacterial test
# patients 19, 20, 36, 42, 86 are used as COVID-19 viral test

for key in filename_label.keys():
    arr = np.array(filename_label[key])
    if arr.size == 0:
        continue
    # split by patients
    # num_diff_patients = len(np.unique(arr[:,0]))
    # num_test = max(1, round(split*num_diff_patients))
    # select num_test number of random patients
    if key == 'pneumonia':
        test_patients = ['8', '31']
    elif key == 'COVID-19':
        test_patients = ['19', '20', '36', '42', '86'] # random.sample(list(arr[:,0]), num_test)
    else: 
        test_patients = []
    print('Key: ', key)
    print('Test patients: ', test_patients)
    # go through all the patients
    for patient in arr:
        if patient[0] in test_patients:
            if (COPY_FILE):
                copyfile(os.path.join(imgpath, patient[1]), os.path.join(savepath, 'test', patient[1]))
                test.append(patient)
                test_count[patient[2]] += 1
            else:
                print("WARNING   :   passing copy file !!!!!!!!!!!!!!!!!!!!!!")
                break
        else:
            if (COPY_FILE):
                copyfile(os.path.join(imgpath, patient[1]), os.path.join(savepath, 'train', patient[1]))
                train.append(patient)
                train_count[patient[2]] += 1
            else:
                print("WARNING   :   passing copy file !!!!!!!!!!!!!!!!!!!!!!")
                break

print('test count: ', test_count)
print('train count: ', train_count)

# ## Preprocess data
# Copy kaggle dataset into train and test folders
# add normal and rest of pneumonia cases from https://www.kaggle.com/c/rsna-pneumonia-detection-challenge

print(kaggle_datapath)
csv_normal = pd.read_csv(os.path.join(kaggle_datapath, kaggle_csvname), nrows=None)
csv_pneu = pd.read_csv(os.path.join(kaggle_datapath, kaggle_csvname2), nrows=None)
patients = {'normal': [], 'pneumonia': []}

for index, row in csv_normal.iterrows():
    if row['class'] == 'Normal':
        patients['normal'].append(row['patientId'])

for index, row in csv_pneu.iterrows():
    if int(row['Target']) == 1:
        patients['pneumonia'].append(row['patientId'])

for key in patients.keys():
    arr = np.array(patients[key])
    if arr.size == 0:
        continue
    # split by patients 
    # num_diff_patients = len(np.unique(arr))
    # num_test = max(1, round(split*num_diff_patients))
    #'/content/COVID-Net/'
    test_patients = np.load('rsna_test_patients_{}.npy'.format(key)) # random.sample(list(arr), num_test)
    # np.save('rsna_test_patients_{}.npy'.format(key), np.array(test_patients))
    for patient in arr:
        ds = dicom.dcmread(os.path.join(kaggle_datapath, kaggle_imgpath, patient + '.dcm'))
        pixel_array_numpy = ds.pixel_array
        imgname = patient + '.png'
        if patient in test_patients:
            if (COPY_FILE):
                cv2.imwrite(os.path.join(savepath, 'test', imgname), pixel_array_numpy)
                test.append([patient, imgname, key])
                test_count[key] += 1
            else:
                print("WARNING   :   passing copy file !!!!!!!!!!!!!!!!!!!!!!")
                break
        else:
            if (COPY_FILE):
                cv2.imwrite(os.path.join(savepath, 'train', imgname), pixel_array_numpy)
                train.append([patient, imgname, key])
                train_count[key] += 1
            else:
                print("WARNING   :   passing copy file !!!!!!!!!!!!!!!!!!!!!!")
                break
print('test count: ', test_count)
print('train count: ', train_count)

# %% [markdown]
# ## Final data stats

# %%
# final stats
print('Final stats')
print('Train count: ', train_count)
print('Test count: ', test_count)
print('Total length of train: ', len(train))
print('Total length of test: ', len(test))

# %% [markdown]
# ## train and test file extraction

# %%
# export to train and test csv
# format as patientid, filename, label, separated by a space
train_file = open("train_split_v2.txt","w") 
for sample in train:
    info = str(sample[0]) + ' ' + sample[1] + ' ' + sample[2] + '\n'
    train_file.write(info)

train_file.close()

test_file = open("test_split_v2.txt", "w")
for sample in test:
    info = str(sample[0]) + ' ' + sample[1] + ' ' + sample[2] + '\n'
    test_file.write(info)

test_file.close()

# %% [markdown]
# # Requirements

# %%
#get_ipython().system(' pip install torch torcvision pandas')

# %% [markdown]
# # Training

# %%
#get_ipython().run_line_magic('cd', 'COVIDNet')
#get_ipython().system(" python main.py --dataset_name='COVIDx' --root_path='/content/covid-chestxray-dataset/data'")

