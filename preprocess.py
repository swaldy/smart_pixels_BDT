import numpy as np
import pandas as pd
from pandas import read_csv
import math
import matplotlib.pyplot as plt
import glob

# Global variables
threshold = 0.2 #in units of GeV
noise_threshold = 400
sensor_geom = '50x12P5x150_0fb'
dataset_savedir =  '/eos/user/s/swaldych/smart_pix/labels/preprocess' # for save loc of final datasets

dirtrain = '/eos/user/s/swaldych/smart_pix/labels/'
# /location/of/parquets/smartpixels/dataset_2s/dataset_2s_50x12P5_parquets/unflipped
dftrain = pd.read_parquet(dirtrain+'labels_d16401.parquet')
print(dftrain.head())
print(dftrain.tail())

trainlabels = []
trainrecons = []

iter=0
suffix = 16400
for filepath in glob.iglob(dirtrain+'labels*.parquet'):
        iter+=3
for i in range(int(iter/3)):
        trainlabels.append(pd.read_parquet(dirtrain+'labels_d'+str(suffix+i+1)+'.parquet'))
        trainrecons.append(pd.read_parquet(dirtrain+'recon2D_d'+str(suffix+i+1)+'.parquet'))
        trainlabels_csv = pd.concat(trainlabels, ignore_index=True)
        trainrecons_csv = pd.concat(trainrecons, ignore_index=True)

iter_0, iter_1, iter_2 = 0, 0, 0
iter_rem = 0
for iter, row in trainlabels_csv.iterrows():
    if(abs(row['pt'])>threshold):
        iter_0+=1
    elif(-1*threshold<=row['pt']<0):
        iter_1+=1
    elif(0<row['pt']<=threshold):
        iter_2+=1
    else:
        iter_rem+=1
# output_file.write("iter_0: "+str(iter_0)+"\n")
# output_file.write("iter_1: "+str(iter_1)+"\n")
# output_file.write("iter_2: "+str(iter_2)+"\n")
# output_file.write("iter_rem: "+str(iter_rem)+"\n")

plt.hist(trainlabels_csv['pt'], bins=100)
plt.title('pT of all events')
plt.savefig(dataset_savedir+"train_pt_all_"+sensor_geom+".png")
plt.close()

plt.hist(trainlabels_csv[abs(trainlabels_csv['pt'])>threshold]['pt'], bins=100)
plt.title('pT of Class 0 events')
plt.savefig(dataset_savedir+"train_pt_cls0_"+sensor_geom+".png")
plt.close()

plt.hist(trainlabels_csv[(0<=trainlabels_csv['pt'])&(trainlabels_csv['pt']<=threshold)]['pt'], bins=50)
plt.hist(trainlabels_csv[(-1*threshold<=trainlabels_csv['pt'])& (trainlabels_csv['pt']<0)]['pt'], bins=50)
plt.title('pT of Class 1+2 events')
plt.savefig(dataset_savedir+"train_pt_cls12_"+sensor_geom+".png")
plt.close()

number_of_events = (min(iter_1, iter_2)//1000)*1000
if(number_of_events*2>iter_0):
    number_of_events = (iter_0//1000)*1000/2
number_of_events = int(number_of_events)
#output_file.write("Number of events: "+str(number_of_events)+"\n")

def sumRow(X):
    X = np.where(X < noise_threshold, 0, X)
    sum1 = 0
    sumList = []
    for i in X:
        sum1 = np.sum(i,axis=0)
        sumList.append(sum1)
        b = np.array(sumList)
    return b
trainlist1, trainlist2 = [], []
hist_temp=[]
for (index1, row1), (index2, row2) in zip(trainrecons_csv.iterrows(), trainlabels_csv.iterrows()):
    rowSum = 0.0
    X = row1.values
    X = np.reshape(X,(13,21))
    rowSum = sumRow(X)
    hist_temp.append(np.sum(rowSum>0))
    trainlist1.append(rowSum)
    cls = -1
    if(abs(row2['pt'])>threshold):
        cls=0
    elif(-1*threshold<=row2['pt']<0):
        cls=1
    elif(0<=row2['pt']<=threshold):
        cls=2
    trainlist2.append([row2['y-local'], cls, row2['pt']])

plt.hist(hist_temp, bins=14,  range=[0, 14], histtype='step', fill=False, density=True)
plt.savefig(dataset_savedir+"y_profile_afterThreshold_"+sensor_geom+".png")
plt.close()

traindf_all = pd.concat([pd.DataFrame(trainlist1), pd.DataFrame(trainlist2 , columns=['y-local', 'cls', 'pt'])], axis=1)

totalsize = number_of_events
random_seed0 = 10#11
random_seed1 = 13#14
random_seed2 = 19#20

traindf_all = traindf_all.sample(frac=1, random_state=random_seed0).reset_index(drop=True)
# traindf_all.to_csv(dataset_savedir+'/'+'/FullTrainData_'+sensor_geom+'_0P'+str(threshold - int(threshold))[2:]+'thresh.csv', index=False)
traindfcls0 = traindf_all.loc[traindf_all['cls']==0]
traindfcls1 = traindf_all.loc[traindf_all['cls']==1]
traindfcls2 = traindf_all.loc[traindf_all['cls']==2]
# output_file.write(str(traindfcls0.shape)+"\n")
# output_file.write(str(traindfcls1.shape)+"\n")
# output_file.write(str(traindfcls2.shape)+"\n")
traindfcls0 = traindfcls0.iloc[:2*totalsize]
traindfcls1 = traindfcls1.iloc[:totalsize]
traindfcls2 = traindfcls2.iloc[:totalsize]

traincls0 = traindfcls0.sample(frac = 1, random_state=random_seed1)
traincls1 = traindfcls1.sample(frac = 1, random_state=random_seed1)
traincls2 = traindfcls2.sample(frac = 1, random_state=random_seed1)
train = pd.concat([traincls0, traincls1, traincls2], axis=0)

train = train.sample(frac=1, random_state=random_seed2)

# output_file.write(str(traincls0.shape)+"\n")
# output_file.write(str(traincls1.shape)+"\n")
# output_file.write(str(traincls2.shape)+"\n")
# output_file.write(str(train.shape)+"\n")

trainlabel = train['cls']
trainpt = train['pt']
train = train.drop(['cls', 'pt'], axis=1)

# output_file.write(str(train.shape)+"\n")
# output_file.write(str(trainlabel.shape)+"\n")
# output_file.write(str(trainpt.shape)+"\n")

train.to_csv(dataset_savedir+'/FullPrecisionInputTrainSet_'+sensor_geom+'_0P'+str(threshold - int(threshold))[2:]+'thresh.csv', index=False)
trainlabel.to_csv(dataset_savedir+'/TrainSetLabel_'+sensor_geom+'_0P'+str(threshold - int(threshold))[2:]+'thresh.csv', index=False)
trainpt.to_csv(dataset_savedir+'/TrainSetPt_'+sensor_geom+'_0P'+str(threshold - int(threshold))[2:]+'thresh.csv', index=False)
