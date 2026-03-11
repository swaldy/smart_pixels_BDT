import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, classification_report
import pandas as pd
import numpy as np
import conifer
import conifer.converters as C
#print([name for name in dir(C) if "convert" in name])
import plotting
import matplotlib.pyplot as plt
from matplotlib.legend import Legend
from matplotlib.lines import Line2D


models_dir = 'root://cmseos.fnal.gov//store/user/swaldych/smart_pix/labels/models'
results_dir = 'root://cmseos.fnal.gov//store/user/swaldych/smart_pix/labels/results'

sensor_geom = "50x12P5x150_0fb"
threshold = 0.2 #in GeV
tag = f"{sensor_geom}_0P{str(threshold - int(threshold))[2:]}thresh"

dfx = pd.read_csv(f"root://cmseos.fnal.gov//store/user/swaldych/smart_pix/labels/preprocess/FullPrecisionInputTrainSet_{tag}.csv") #y-local
dfy = pd.read_csv(f"root://cmseos.fnal.gov//store/user/swaldych/smart_pix/labels/preprocess/TrainSetLabel_{tag}.csv") 
pt=pd.read_csv(f"root://cmseos.fnal.gov//store/user/swaldych/smart_pix/labels/preprocess/TrainSetPt_{tag}.csv")

X = dfx.values
y = dfy.values
real_pt=pt.values

X_train, X_test, y_train, y_test, pt_train, pt_test = train_test_split(
    X, y, real_pt, test_size=0.2, shuffle=True, random_state=42) #we are saying split arrays the same way

#  Convert data to XGBoost DMatrix format
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    "objective": "multi:softprob",  # "softmax option produces a trained Booster object whose predict method returns a 1d array of predicted labels, whereas the softprob option produces a trained Booster object whose predict method returns a 2d array of predicted probabilities"
    "eval_metric": "auc",  # AUC for evaluation
    "max_depth": 5,  # Tree depth
    "eta": 0.05,  # Learning rate
    "seed": 42,  # Reproducibility
    "num_class":3 #three classes...0,1,2 
}

model = xgb.train(
    params,
    dtrain,
    num_boost_round=200,
    evals=[(dtrain, "train"), (dtest, "test")],
    early_stopping_rounds=10,
    verbose_eval=10  # Print progress every 10 iterations
)

#Covert model to FPGA firmware with `conifer`
cfg = conifer.backends.xilinxhls.auto_config()

# print the config
print('Default Configuration\n' + '-' * 50)
plotting.print_dict(cfg)
print('-' * 50)

# modify the config
cfg['OutputDir'] = 'root://cmseos.fnal.gov//store/user/swaldych/smart_pix/labels/generated_firmware_files' #where to put all generated firmware files
cfg['XilinxPart'] = 'xcu250-figd2104-2L-e' #the part number for an FPGA. Taken from example (Alveo U50)

# print the config again
print('Modified Configuration\n' + '-' * 50)
plotting.print_dict(cfg)
print('-' * 50)

# convert the model to the conifer representation
conifer_model = conifer.converters.convert_from_xgboost(model, cfg)
# write the project (writing HLS project to disk)
conifer_model.write()
model.save_model(f"{models_dir}/xgboost_model_conifer_{tag}.json")

conifer_model.compile()
y_hls = conifer_model.decision_function(X_test)

y_hls_proba = softmax(y_hls)  # compute class probabilities from the raw predictions

print(f'Accuracy baseline:  {accuracy_score(np.argmax(y_test_one_hot, axis=1), np.argmax(y_ref, axis=1)):.5f}')
print(f'Accuracy xgboost:   {accuracy_score(np.argmax(y_test_one_hot, axis=1), np.argmax(y_xgb, axis=1)):.5f}')
print(f'Accuracy conifer:   {accuracy_score(np.argmax(y_test_one_hot, axis=1), np.argmax(y_hls_proba, axis=1)):.5f}')


fig, ax = plt.subplots(figsize=(9, 9))
_ = plotting.makeRoc(y_test_one_hot, y_ref, classes, linestyle='--')
plt.gca().set_prop_cycle(None)  # reset the colors
_ = plotting.makeRoc(y_test_one_hot, y_xgb, classes, linestyle=':')
plt.gca().set_prop_cycle(None)  # reset the colors
_ = plotting.makeRoc(y_test_one_hot, y_hls_proba, classes, linestyle='-')

# add a legend

lines = [
    Line2D([0], [0], ls='--'),
    Line2D([0], [0], ls=':'),
    Line2D([0], [0], ls='-'),
]

leg = Legend(ax, lines, labels=['part1 Keras', 'xgboost', 'conifer'], loc='lower right', frameon=False)
ax.add_artist(leg)
