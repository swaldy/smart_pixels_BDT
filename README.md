This repo was heavily inspired by both https://github.com/smart-pix/filter and https://github.com/fatimargz/Locked-in-Leptons/tree/main

To clone this repo do...
```bash
git clone --recursive https://github.com/swaldy/smart_pixels_BDT.git
```
I run all code on lxplus. Before running any scripts, in a clean shell do...

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_106a/x86_64-el9-gcc11-opt/setup.sh
python3 -m venv --system-site-packages venvs/hls4ml_conifer
source venvs/hls4ml_conifer/bin/activate
pip install --upgrade pip
pip install hls4ml conifer
```
After you do this the first time, all other times you log on, execute these steps...
```
source /cvmfs/sft.cern.ch/lcg/views/LCG_106a/x86_64-el9-gcc11-opt/setup.sh
source venvs/hls4ml_conifer/bin/activate
```
