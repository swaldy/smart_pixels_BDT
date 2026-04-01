
import numpy as np
import pandas as pd
from pandas import read_csv
import math
import matplotlib.pyplot as plt
import glob
from copy import deepcopy
import sys
import os

# Global variables
threshold = 0.2
pixel_arrayX = 21 #16 #
pixel_arrayY = 13 #16 #
noise_threshold = 0
qm_charge_levels = [400, 1600, 2400]
qm_quant_values = [0,1,2,3]
y_local_bins = np.linspace(-8.1, 8.1, 13)
bin_number = 6
y_local_min, y_local_max = y_local_bins[bin_number], y_local_bins[bin_number+1]
sensor_geom = '50x12P5'
train_dataset_name = '/uscms/home/swaldych/nobackup/dataset_3s' # for train datasets
test_dataset_name = '/uscms/home/swaldych/nobackup/dataset_2s' # for location of test (physical pT) datasets
dataset_savedir = f'dataset_3_{noise_threshold}NoiseThresh' # for save loc of final datasets
parquet_suffix = 16400

# sensor_geom = '50x12P5_100e-sigma'
# train_dataset_name = 'dataset_3sNoise_16x16' # for train datasets
# test_dataset_name = 'dataset_2sNoise_16x16' # for location of test (physical pT) datasets
# dataset_savedir = f'dataset_2sNoise_16x16_50x12P5_100e-sigma_{noise_threshold}NoiseThresh' # for save loc of final datasets
if not os.path.exists(dataset_savedir):
    os.makedirs(dataset_savedir)

def sumRow(X):
        X = np.where(X < noise_threshold, 0, X)
        sum1 = 0
        sumList = []
        for i in X:
            sum1 = np.sum(i,axis=0)
            sumList.append(sum1)
            b = np.array(sumList)
        return b
  
dirtrain = train_dataset_name+'/'+'dataset_3s'+'_'+sensor_geom+'_parquets/unflipped/'
print(dirtrain)

trainlabels = []
trainrecons = []

iter=0
for filepath in glob.iglob(dirtrain+'labels*.parquet'):
    iter+=3
print(iter," files present in directory.")
for i in range(int(iter/3)):
        trainlabels.append(pd.read_parquet(dirtrain+'labels_d'+str(parquet_suffix+i+1)+'.parquet'))
        trainrecons.append(pd.read_parquet(dirtrain+'recon2D_d'+str(parquet_suffix+i+1)+'.parquet'))
trainlabels_csv = pd.concat(trainlabels, ignore_index=True)
trainrecons_csv = pd.concat(trainrecons, ignore_index=True)
# Retain events from only one y-local bin
filtered_indices = trainlabels_csv[(trainlabels_csv['y-local'] >= y_local_min) & (trainlabels_csv['y-local'] < y_local_max)].index
trainlabels_csv = trainlabels_csv.loc[filtered_indices].reset_index(drop=True)
trainrecons_csv = trainrecons_csv.loc[filtered_indices].reset_index(drop=True)

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
print("iter_0: ",iter_0)
print("iter_1: ",iter_1)
print("iter_2: ",iter_2)
print("iter_rem: ",iter_rem)

# plt.hist(trainlabels_csv['pt'], bins=100)
# plt.title('pT of all events')
# plt.show()

# plt.hist(trainlabels_csv[abs(trainlabels_csv['pt'])>threshold]['pt'], bins=100)
# plt.title('pT of Class 0 events')
# plt.show()

# plt.hist(trainlabels_csv[(0<=trainlabels_csv['pt'])&(trainlabels_csv['pt']<=threshold)]['pt'], bins=50)
# plt.hist(trainlabels_csv[(-1*threshold<=trainlabels_csv['pt'])& (trainlabels_csv['pt']<0)]['pt'], bins=50)
# plt.title('pT of Class 1+2 events')
# plt.show()

number_of_events = (min(iter_1, iter_2)//1000)*1000
if(number_of_events*2>iter_0):
    number_of_events = (iter_0//1000)*1000/2
number_of_events = int(number_of_events)
print("Number of events: ",number_of_events)

def quantize_manual(x, 
                    charge_levels=[400,800,1200],
                    quant_values=[0,1,2,3],
                    shuffled=True
                   ):
    '''
    Quantize a df with manually defined charge level boundaries
        x (np.array or pd.DataFrame): input data (with or without labels).
        charge_levels (list, shape=(N-1)): finite charge levels for boundaries of N bins.
            eg. for N=4 bins with boundaries [-9e19, 400], [400, 800], [800,1200], [1200, 9e19]
            use: charge_levels = [400, 800, 1200]
        quant_values (list): list of values for each of N charge bins
        shuffled (bool, default: True): is this dataframe from a dataset shuffled? ie. are
            clusters and labels both present in the dataframe x
    '''
    # start_time = time.time()
    df = deepcopy(x)
    if shuffled:
        cols = [c for c in df.columns if c.isnumeric()]
        data = df[cols].values
    else:
        data = df.values
    # print(f'get data: {time.time()-start_time:.4f}', f'total: {time.time()-start_time:.4f}')
    # newtime = time.time()
    charge_levels= np.array(charge_levels)
    minval, maxval = [-9e19], [9e19]

    #pad the charge levels with +/- inf
    charge_levels = np.append(minval, charge_levels)
    charge_levels= np.append(charge_levels, maxval)

    #turn charge_levels into bin boundaries
    bins = None
    for c in range(len(charge_levels)-1):
        if bins is None:
            bins = [[charge_levels[c], charge_levels[c+1]]]
        else:
            bins = np.append(bins, [[charge_levels[c], charge_levels[c+1]]], axis =0)
    # print(f'make bins: {time.time()-newtime:.4f}', f'total: {time.time()-start_time:.4f}')
    # newtime = time.time()

    #quantize the data
    dfq = pd.DataFrame(np.zeros_like(data),index=df.index, columns=cols)
    for j, binbounds in enumerate(bins):
        #mask pixels by charge bin
        mask = (data>binbounds[0]) & (data<binbounds[1])
        dfq = dfq.mask(mask, quant_values[j])
    if shuffled:
        df[cols] = dfq
    else:
        df = dfq
    # print(f'make quantized data: {time.time()-newtime:.4f}', f'total: {time.time()-start_time:.4f}')

    try:
        return df
    finally:
        del dfq, data, mask, df, cols

trainrecons_csv.head()
trainrecons_csv_quantized = quantize_manual(trainrecons_csv, charge_levels=qm_charge_levels, quant_values=qm_quant_values, shuffled=True)
pd.options.display.max_columns = None
trainrecons_csv_quantized.head()

trainlist1, trainlist2 = [], []
hist_temp=[]
for (index1, row1), (index2, row2) in zip(trainrecons_csv_quantized.iterrows(), trainlabels_csv.iterrows()):
    rowSum = 0.0
    X = row1.values
    X = np.reshape(X,(pixel_arrayY,pixel_arrayX))
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
    trainlist2.append([cls, row2['pt']])
    # trainlist2.append([row2['y-local'], cls, row2['pt']]) # y-local is not passed in ASIC DNN as there is one DNN per y-local bin.

# plt.hist(hist_temp, bins=14,  range=[0, 14], histtype='step', fill=False, density=True)
# plt.show()
traindf_all = pd.concat([pd.DataFrame(trainlist1), pd.DataFrame(trainlist2 , columns=['cls', 'pt'])], axis=1)
# traindf_all = pd.concat([pd.DataFrame(trainlist1), pd.DataFrame(trainlist2 , columns=['y-local', 'cls', 'pt'])], axis=1)
print(traindf_all.head())

totalsize = number_of_events
random_seed0 = 10#11
random_seed1 = 13#14
random_seed2 = 19#20

traindf_all = traindf_all.sample(frac=1, random_state=random_seed0).reset_index(drop=True)
# traindf_all.to_csv(dataset_savedir+'/'+'/FullTrainData_'+sensor_geom+'_0P'+str(threshold - int(threshold))[2:]+'thresh.csv', index=False)
traindfcls0 = traindf_all.loc[traindf_all['cls']==0]
traindfcls1 = traindf_all.loc[traindf_all['cls']==1]
traindfcls2 = traindf_all.loc[traindf_all['cls']==2]
print(traindfcls0.shape)
print(traindfcls1.shape)
print(traindfcls2.shape)
print(traindfcls2.head())
# don't create balanced dataset as only 2000 events present in class 1/2 for 6th y-local bin
traindfcls0 = traindfcls0.iloc[:2*totalsize]
traindfcls1 = traindfcls1.iloc[:totalsize]
traindfcls2 = traindfcls2.iloc[:totalsize]
print(traindfcls2.head())

traincls0 = traindfcls0.sample(frac = 1, random_state=random_seed1)
traincls1 = traindfcls1.sample(frac = 1, random_state=random_seed1)
traincls2 = traindfcls2.sample(frac = 1, random_state=random_seed1)
train = pd.concat([traincls0, traincls1, traincls2], axis=0)

train = train.sample(frac=1, random_state=random_seed2)

print(traincls0.shape)
print(traincls1.shape)
print(traincls2.shape)
print(train.shape)

trainlabel = train['cls']
trainpt = train['pt']
train = train.drop(['cls', 'pt'], axis=1)

print(train.shape)
print(trainlabel.shape)
print(trainpt.shape)

train.to_csv(dataset_savedir+'/QuantizedInputTrainSet_'+sensor_geom+'_0P'+str(threshold - int(threshold))[2:]+'thresh.csv', index=False)
trainlabel.to_csv(dataset_savedir+'/QuantizedTrainSetLabel_'+sensor_geom+'_0P'+str(threshold - int(threshold))[2:]+'thresh.csv', index=False)
trainpt.to_csv(dataset_savedir+'/QuantizedTrainSetPt_'+sensor_geom+'_0P'+str(threshold - int(threshold))[2:]+'thresh.csv', index=False)

dirtest = test_dataset_name+'/'+train_dataset_name+'_'+sensor_geom+'_parquets/unflipped/'
# dirtest = '/asic/projects/C/CMS_PIX_28/pixelAV_datasets/unshuffled_DO_NOT_DELETE/'+test_dataset_name+'/'+test_dataset_name+'_'+sensor_geom+'_parquets/unflipped/'
# /location/of/parquets/smartpixels/dataset_2s/dataset_2s_50x12P5_parquets/unflipped
dftest = pd.read_parquet(f'{dirtest}labels_d{parquet_suffix+1}.parquet')
print(dftest.head())
print(dftest.tail())

testlabels = []
testrecons = []

iter=0
for filepath in glob.iglob(dirtest+'labels*.parquet'):
    iter+=3
print(iter," files present in directory.")
for i in range(int(iter/3)):
        testlabels.append(pd.read_parquet(dirtest+'labels_d'+str(parquet_suffix+i+1)+'.parquet'))
        testrecons.append(pd.read_parquet(dirtest+'recon2D_d'+str(parquet_suffix+i+1)+'.parquet'))
testlabels_csv = pd.concat(testlabels, ignore_index=True)
testrecons_csv = pd.concat(testrecons, ignore_index=True)

# Retain events from only one y-local bin
filtered_indices = testlabels_csv[(testlabels_csv['y-local'] >= y_local_min) & (testlabels_csv['y-local'] < y_local_max)].index
testlabels_csv = testlabels_csv.loc[filtered_indices].reset_index(drop=True)
testrecons_csv = testrecons_csv.loc[filtered_indices].reset_index(drop=True)

iter_0, iter_1, iter_2 = 0, 0, 0
iter_rem = 0
for iter, row in testlabels_csv.iterrows():
    if(abs(row['pt'])>threshold):
        iter_0+=1
    elif(-1*threshold<=row['pt']<0):
        iter_1+=1
    elif(0<row['pt']<=threshold):
        iter_2+=1
    else:
        iter_rem+=1
print("iter_0: ",iter_0)
print("iter_1: ",iter_1)
print("iter_2: ",iter_2)
print("iter_rem: ",iter_rem)

# plt.hist(testlabels_csv['pt'], bins=100)
# plt.title('pT of all events')
# plt.show()

# plt.hist(testlabels_csv[abs(testlabels_csv['pt'])>threshold]['pt'], bins=100)
# plt.title('pT of Class 0 events')
# plt.show()

# plt.hist(testlabels_csv[(0<=testlabels_csv['pt'])&(testlabels_csv['pt']<=threshold)]['pt'], bins=50)
# plt.hist(testlabels_csv[(-1*threshold<=testlabels_csv['pt'])& (testlabels_csv['pt']<0)]['pt'], bins=50)
# plt.title('pT of Class 1+2 events')
# plt.show()

number_of_events = (min(iter_1, iter_2)//1000)*1000
if(number_of_events*2>iter_0):
    number_of_events = (iter_0//1000)*1000/2
number_of_events = int(number_of_events)
print("Number of events: ",number_of_events)

testrecons_csv.head()
testrecons_csv_quantized = quantize_manual(testrecons_csv, charge_levels=qm_charge_levels, quant_values=qm_quant_values, shuffled=True)
pd.options.display.max_columns = None
testrecons_csv_quantized.head()

testlist1, testlist2 = [], []

for (index1, row1), (index2, row2) in zip(testrecons_csv_quantized.iterrows(), testlabels_csv.iterrows()):
    rowSum = 0.0
    X = row1.values
    X = np.reshape(X,(pixel_arrayY,pixel_arrayX))
    rowSum = sumRow(X)
    testlist1.append(rowSum)
    cls = -1
    if(abs(row2['pt'])>threshold):
        cls=0
    elif(-1*threshold<=row2['pt']<0):
        cls=1
    elif(0<=row2['pt']<=threshold):
        cls=2
    testlist2.append([cls, row2['pt']])
testdf_all = pd.concat([pd.DataFrame(testlist1), pd.DataFrame(testlist2 , columns=['cls', 'pt'])], axis=1)
print(testdf_all.head())

# totalsize = number_of_events#227000
random_seed0 = 10#11
random_seed1 = 13#14
random_seed2 = 19#20

testdf_all = testdf_all.sample(frac=1, random_state=random_seed0).reset_index(drop=True)
testdf_all.to_csv(dataset_savedir+'/'+'/FullTestData_'+sensor_geom+'_0P'+str(threshold - int(threshold))[2:]+'thresh.csv', index=False)
# testdfcls0 = testdf_all.loc[testdf_all['cls']==0]
# testdfcls1 = testdf_all.loc[testdf_all['cls']==1]
# testdfcls2 = testdf_all.loc[testdf_all['cls']==2]
# print(testdfcls0.shape)
# print(testdfcls1.shape)
# print(testdfcls2.shape)
# print(testdfcls2.head())
# testdfcls0 = testdfcls0.iloc[:2*totalsize]
# testdfcls1 = testdfcls1.iloc[:totalsize]
# testdfcls2 = testdfcls2.iloc[:totalsize]
# print(testdfcls2.head())

# testcls0 = testdfcls0.sample(frac = 1, random_state=random_seed1)
# testcls1 = testdfcls1.sample(frac = 1, random_state=random_seed1)
# testcls2 = testdfcls2.sample(frac = 1, random_state=random_seed1)
# test = pd.concat([testcls0, testcls1, testcls2], axis=0)

# test = test.sample(frac=1, random_state=random_seed2)
test=testdf_all
# print(testcls0.shape)
# print(testcls1.shape)
# print(testcls2.shape)
print(test.shape)

testlabel = test['cls']
testpt = test['pt']
test = test.drop(['cls', 'pt'], axis=1)

print(test.shape)
print(testlabel.shape)
print(testpt.shape)

test.to_csv(dataset_savedir+'/QuantizedInputTestSet_'+sensor_geom+'_0P'+str(threshold - int(threshold))[2:]+'thresh.csv', index=False)
testlabel.to_csv(dataset_savedir+'/QuantizedTestSetLabel_'+sensor_geom+'_0P'+str(threshold - int(threshold))[2:]+'thresh.csv', index=False)
testpt.to_csv(dataset_savedir+'/QuantizedTestSetPt_'+sensor_geom+'_0P'+str(threshold - int(threshold))[2:]+'thresh.csv', index=False)
