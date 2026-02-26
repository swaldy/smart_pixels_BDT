import numpy as np
import pandas as pd
from pandas import read_csv
import matplotlib.pyplot as plt

# -------------------------
# Global variables
# -------------------------
threshold = 0.2
noise_threshold = 400
sensor_geom = '50x12P5x150_0fb'
dataset_savedir = '/eos/user/s/swaldych/smart_pix/labels/preprocess/'
dirtrain = '/eos/user/s/swaldych/smart_pix/labels/'

# -------------------------
# Manually define parquet range
# -------------------------
start_id = 16401
end_id   = 16441   # <-- set this to your last dataset

trainlabels = []
trainrecons = []

for idx in range(start_id, end_id + 1):
    trainlabels.append(pd.read_parquet(dirtrain + f'labels_d{idx}.parquet'))
    trainrecons.append(pd.read_parquet(dirtrain + f'recon2D_d{idx}.parquet'))

trainlabels_csv = pd.concat(trainlabels, ignore_index=True)
trainrecons_csv = pd.concat(trainrecons, ignore_index=True)

# -------------------------
# Count classes
# -------------------------
iter_0, iter_1, iter_2, iter_rem = 0, 0, 0, 0

for _, row in trainlabels_csv.iterrows():
    pt = row['pt']
    if abs(pt) > threshold:
        iter_0 += 1
    elif -threshold <= pt < 0:
        iter_1 += 1
    elif 0 <= pt <= threshold:
        iter_2 += 1
    else:
        iter_rem += 1

# -------------------------
# Histograms
# -------------------------
plt.hist(trainlabels_csv['pt'], bins=100)
plt.title('pT of all events')
plt.savefig(dataset_savedir + "train_pt_all_" + sensor_geom + ".png")
plt.close()

plt.hist(trainlabels_csv[abs(trainlabels_csv['pt']) > threshold]['pt'], bins=100)
plt.title('pT of Class 0 events')
plt.savefig(dataset_savedir + "train_pt_cls0_" + sensor_geom + ".png")
plt.close()

plt.hist(trainlabels_csv[(0 <= trainlabels_csv['pt']) &
                         (trainlabels_csv['pt'] <= threshold)]['pt'], bins=50)
plt.hist(trainlabels_csv[(-threshold <= trainlabels_csv['pt']) &
                         (trainlabels_csv['pt'] < 0)]['pt'], bins=50)
plt.title('pT of Class 1+2 events')
plt.savefig(dataset_savedir + "train_pt_cls12_" + sensor_geom + ".png")
plt.close()

# -------------------------
# Balance dataset
# -------------------------
number_of_events = (min(iter_1, iter_2) // 1000) * 1000
if number_of_events * 2 > iter_0:
    number_of_events = (iter_0 // 1000) * 1000 // 2

number_of_events = int(number_of_events)

# -------------------------
# Feature extraction
# -------------------------
def sumRow(X):
    X = np.where(X < noise_threshold, 0, X)
    return np.sum(X, axis=0)

trainlist1 = []
trainlist2 = []
hist_temp = []

for (index1, row1), (index2, row2) in zip(trainrecons_csv.iterrows(),
                                          trainlabels_csv.iterrows()):

    X = row1.values.reshape(13, 21)
    rowSum = sumRow(X)

    hist_temp.append(np.sum(rowSum > 0))
    trainlist1.append(rowSum)

    if abs(row2['pt']) > threshold:
        cls = 0
    elif -threshold <= row2['pt'] < 0:
        cls = 1
    else:
        cls = 2

    trainlist2.append([row2['y-local'], cls, row2['pt']])

plt.hist(hist_temp, bins=14, range=[0, 14], histtype='step', density=True)
plt.savefig(dataset_savedir + "y_profile_afterThreshold_" + sensor_geom + ".png")
plt.close()

# -------------------------
# Final dataframe
# -------------------------
traindf_all = pd.concat(
    [pd.DataFrame(trainlist1),
     pd.DataFrame(trainlist2, columns=['y-local', 'cls', 'pt'])],
    axis=1
)

totalsize = number_of_events

traindf_all = traindf_all.sample(frac=1, random_state=10).reset_index(drop=True)

traindfcls0 = traindf_all[traindf_all['cls'] == 0].iloc[:2*totalsize]
traindfcls1 = traindf_all[traindf_all['cls'] == 1].iloc[:totalsize]
traindfcls2 = traindf_all[traindf_all['cls'] == 2].iloc[:totalsize]

train = pd.concat([traindfcls0, traindfcls1, traindfcls2], axis=0)
train = train.sample(frac=1, random_state=19)

trainlabel = train['cls']
trainpt = train['pt']
train = train.drop(['cls', 'pt'], axis=1)

train.to_csv(dataset_savedir + 'FullPrecisionInputTrainSet_' +
             sensor_geom + '_0P2thresh.csv', index=False)

trainlabel.to_csv(dataset_savedir + 'TrainSetLabel_' +
                  sensor_geom + '_0P2thresh.csv', index=False)

trainpt.to_csv(dataset_savedir + 'TrainSetPt_' +
               sensor_geom + '_0P2thresh.csv', index=False)
