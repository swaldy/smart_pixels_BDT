This repo was heavily inspired by https://github.com/smart-pix/filter and https://github.com/fatimargz/Locked-in-Leptons/tree/main and https://hub.gesis.mybinder.org/user/fastmachinelear-hls4ml-tutorial-w3idccg7/lab

To clone this repo do...
```bash
git clone --recursive https://github.com/swaldy/smart_pixels_BDT.git
```
**note**: If working on LPC CAF (recommended for further steps with HLS4ML), please clone on ```nobackup```, NOT your home. This way you have enough disk space!
## Files
preprocess.py -- takes label and recon2D .parquet files and gets them ready for training. Saves files in CERNBox with path hardcoded

BDT -- contains scripts related to training BDT

NN -- contains scripts related to training NN
# ------LPC CAF (FNAL) SETUP EXAMPLES (recommended)-----------
The steps below describe how to run scripts when working on LPC CAF resources. Guide on how to set up LPC with eos is found here: https://www.uscms.org/uscms_at_work/physics/computing/getstarted/uaf.shtml

## Step 0: set up
If needed, move any files from CERN eos to FNAL eos (must have up to date grid cert! ikf you need to renew go here: https://ca.cern.ch/ca/user/Request.aspx?template=ee2user)
...
```
scp /path/on/your/laptop/YourCert.p12 user@cmslpc-el9.fnal.gov:~/.globus/usercred.p12
ssh user@cmslpc-el9.fnal.gov
chmod 600 ~/.globus/usercred.p12
voms-proxy-init -voms cms

#replace path with yours...
xrdcp -r \
root://eosuser.cern.ch//eos/user/s/swaldych/smart_pix \
root://cmseos.fnal.gov//store/user/swaldych/
```
Now lets set up a place to get work done: 
```
mkdir -p ~/nobackup/myenvs
cd ~/nobackup/myenvs
python3 -m venv lpc-ml
source lpc-ml/bin/activate
pip install conifer xgboost==1.7.6 scikit-learn numpy scipy matplotlib pandas hls4ml #this took about 8 minutes for me
pip install pyarrow
```
# ------LXPLUS SETUP EXAMPLES (Not recommended)-----------

The steps below describe how to run scripts when working in lxplus9

## Step 0: set up python venv

I run all code on lxplus9. Before running any scripts, in a clean shell do...

```bash
python3 -m venv --system-site-packages venvs/hls4ml_conifer_clean
source venvs/hls4ml_conifer_clean/bin/activate
pip install conifer xgboost==1.7.6 scikit-learn numpy scipy matplotlib pandas hls4ml
pip install pyarrow
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
