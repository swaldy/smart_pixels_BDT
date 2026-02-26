from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn import datasets, svm, metrics
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import load_model
# import keras
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import read_csv
import math
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split


sensor_geom = "50x12P5x150_0fb"
threshold = 0.2 #in GeV
seed = 13  
results_dir = '/eos/user/s/swaldych/smart_pix/labels/results'
models_dir = '/eos/user/s/swaldych/smart_pix/labels/models'
tf.random.set_seed(seed)

tag = f"{sensor_geom}_0P{str(threshold - int(threshold))[2:]}thresh"

print("=============================")
print(f"Training model for {sensor_geom} at pT boundary = {threshold}, seed={seed}")

dfX = pd.read_csv(f"/eos/user/s/swaldych/smart_pix/labels/preprocess/FullPrecisionInputTrainSet_{tag}.csv") #y-local
dfy = pd.read_csv(f"/eos/user/s/swaldych/smart_pix/labels/preprocess/TrainSetLabel_{tag}.csv") 
pt=pd.read_csv(f"/eos/user/s/swaldych/smart_pix/labels/preprocess/TrainSetPt_{tag}.csv")

X = dfX.values
y = dfy.values.ravel()
real_pt=pt.values.ravel()

X_train, X_test, y_train, y_test, pt_train, pt_test = train_test_split(
    X, y, real_pt, test_size=0.2, shuffle=True
) #we are saying split arrays the same way

#you cant load in the real pt and then plot is bc the line above shuffles things around. We need the real pt to be shuffled too otherwise its like plotting event A with event B. We need A with A. 

# df1 = pd.read_csv(f"/eos/user/s/swaldych/smart_pix/labels/preprocess/FullPrecisionInputTrainSet_{tag}.csv")
# df2 = pd.read_csv(f"/eos/user/s/swaldych/smart_pix/labels/preprocess/TrainSetLabel_{tag}.csv")
# # df3 = pd.read_csv(f"/eos/user/s/swaldych/smart_pix/labels/preprocess/FullPrecisionInputTestSet_{tag}.csv")
# # df4 = pd.read_csv(f"/eos/user/s/swaldych/smart_pix/labels/preprocess/TestSetLabel_{tag}.csv")

# X_train = df1.values
# X_test  = df3.values
# y_train = df2.values.ravel()  
# y_test  = df4.values.ravel()

# scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

model = tf.keras.models.Sequential([
              tf.keras.layers.Flatten(input_shape=(14,)),
              tf.keras.layers.Dense(128, activation='relu'),
              tf.keras.layers.Dense(3, activation='softmax')
            ])
            
model.compile(optimizer=Adam(learning_rate=0.001),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False), # default from_logits=False
              metrics=[keras.metrics.SparseCategoricalAccuracy()])
            
model.summary()

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=20,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=200,
    batch_size=1024,
    callbacks=[es],
    shuffle=True,
    verbose=1
)

# classes = np.array([0,1,2])
# w = compute_class_weight("balanced", classes=classes, y=y_train.astype(int))
# class_weight = {int(c): float(wi) for c, wi in zip(classes, w)}


history_dict = history.history

print("y counts:", np.bincount(y.astype(int), minlength=3))
print("y_train counts:", np.bincount(y_train.astype(int), minlength=3))
print("y_test counts:", np.bincount(y_test.astype(int), minlength=3))

# --- LOSS ---
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'orange', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f"{results_dir}/loss_{tag}.png")
plt.close()

# --- ACCURACY ---
acc = history_dict['sparse_categorical_accuracy']
val_acc = history_dict['val_sparse_categorical_accuracy']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'orange', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(f"{results_dir}/accuracy_{tag}.png")
plt.close()

# --- PREDICTIONS ---
preds = model.predict(X_test)
print(type(preds))
print(preds)
pred_class = np.argmax(preds, axis=1)

print("pred_class counts:", np.bincount(pred_class, minlength=3))
print("overall acceptance (pred==0):", np.mean(pred_class == 0))

accepted = (pred_class == 0)

pt_vals = []
acc_vals = []

step = 0.2   # GeV
pmin = pt_test.min()
pmax = pt_test.max()

p = pmin
while p < pmax:

    total = 0
    passed = 0

    for i in range(len(pt_test)):
        if p <= pt_test[i] < p + step:
            total += 1
            if accepted[i]:
                passed += 1

    if total > 0:
        pt_vals.append(p + step/2)
        acc_vals.append(passed / total)
        err = np.sqrt(p * (1 - p) / total)

    p += step


plt.errorbar(pt_vals, acc_vals,err,fmt='o',markersize=3)
plt.xlabel("true pt (GeV)")
plt.title("NN Model 2: Classifier acceptance as a function of pT")
plt.ylabel("classifier acceptance pT > |0.2| GeV")
plt.ylim(0,1)
plt.savefig("/eos/user/s/swaldych/smart_pix/labels/models/NN_class_acceptance.png", dpi=300, bbox_inches="tight")
plt.close()

pd.DataFrame(pred_class, columns=["predict"]).to_csv(
    f"{results_dir}/pred_class_NN_{tag}.csv", index=False
)

pd.DataFrame(pt_test, columns=["pt_test"]).to_csv(
    f"{results_dir}/pt_test_NN_{tag}.csv", index=False
)

pd.DataFrame(y_test, columns=["true"]).to_csv(
    f"{results_dir}/testResults_{tag}.csv", index=False
)

# --- TEST METRICS ---
score = model.evaluate(X_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, pred_class)
disp.figure_.suptitle("Multiclassifier Confusion Matrix")
plt.savefig(f"{results_dir}/confusionMatrix_{tag}.png")
plt.close()

# --- SAVE MODEL ---
model.save_weights(f"{models_dir}/trained_model_{tag}.weights.h5")
model.save(f"{models_dir}/trained_model_{tag}.h5")