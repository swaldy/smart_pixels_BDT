This repo was heavily inspired by https://github.com/smart-pix/filter and https://github.com/fatimargz/Locked-in-Leptons/tree/main and https://hub.gesis.mybinder.org/user/fastmachinelear-hls4ml-tutorial-w3idccg7/lab

To clone this repo do...
```bash
git clone --recursive https://github.com/swaldy/smart_pixels_BDT.git
```
## Files
preprocess.py -- takes label and recon2D .parquet files and gets them ready for training. Saves files in CERNBox with path hardcoded

BDT -- contains scripts related to training BDT

NN -- contains scripts related to training NN

## Step 0: set up python venv

I run all code on lxplus9. Before running any scripts, in a clean shell do...

```bash
python3 -m venv --system-site-packages venvs/hls4ml_conifer_clean
source venvs/hls4ml_conifer_clean/bin/activate
pip install conifer xgboost==1.7.6 scikit-learn numpy scipy matplotlib pandas hls4ml
```
After you do this the first time, all other times you log on, execute these steps...
```bash
source venvs/hls4ml_conifer_clean/bin/activate
```

## Step 1: run preprocess.py
make sure you are in environment (hls4ml_conifer_clean). Then run this:
```bash
python3 preprocess.py
```
It will take about 8 minutes. Don't be alarmed. Please note that the file numbers to scan over are hardcoded as well as all paths.

## Step 2: run training
go to either NN or BDT to do training. 

```bash
cd BDT
```
