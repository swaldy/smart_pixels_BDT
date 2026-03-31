%env TF_USE_LEGACY_KERAS 1
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
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

sizes = ['50x12P5']
# sizes = ['50x15']
# dataset_name = 'dataset_7s'
# results_dir = 'results_7s'
# models_dir = 'models_7s'
dataset_name = '/eos/user/s/swaldych/smart_pix/labels/preprocess/'
results_dir = 'results'
models_dir = 'models'
threshold = 0.2
prime_num = [2,3,5,7,11,13,17,19,23,29]
for run_iter in range(10):
    for size_iter in sizes:
        #for threshold in thresholds:
            tf.random.set_seed(prime_num[run_iter])
            sensor_geom = size_iter
            print("=============================")
            print("Run "+str(run_iter)+": Training model for ",sensor_geom," at pT boundary = ",threshold)
            df1 = pd.read_csv(dataset_name+'/QuantizedInputTrainSet_'+sensor_geom+'_0P'+str(threshold - int(threshold))[2:]+'thresh.csv')
            print("Shape of train dataset = ",df1.shape)
            df2 = pd.read_csv(dataset_name+'/QuantizedTrainSetLabel_'+sensor_geom+'_0P'+str(threshold - int(threshold))[2:]+'thresh.csv')
            print("Shape of train-label set = ",df2.shape)
            df3 = pd.read_csv(dataset_name+'/QuantizedInputTestSet_'+sensor_geom+'_0P'+str(threshold - int(threshold))[2:]+'thresh.csv')
            print("Shape of test dataset = ",df3.shape)
            df4 = pd.read_csv(dataset_name+'/QuantizedTestSetLabel_'+sensor_geom+'_0P'+str(threshold - int(threshold))[2:]+'thresh.csv')
            print("Shape of test-label set = ",df4.shape)
            X_train = df1.values
            X_test = df3.values
            y_train = df2.values
            y_test = df4.values
            #X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20, random_state = 0)
            print("X-train, X-test, Y-train, Y-test shapes = ",X_train.shape, X_test.shape, y_train.shape, y_test.shape)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
            X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
            model = tf.keras.models.Sequential([
              tf.keras.layers.Flatten(input_shape=(14,)),
              tf.keras.layers.Dense(128, activation='relu'),
              tf.keras.layers.Dense(3, activation='softmax')
            ])
            
            model.compile(optimizer=Adam(),
                          loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False), # default from_logits=False
                          metrics=[keras.metrics.SparseCategoricalAccuracy()])
            
            model.summary()
            
            es = EarlyStopping(monitor='val_sparse_categorical_accuracy', 
                                               mode='max', # don't minimize the accuracy!
                                               patience=20,
                                               restore_best_weights=True)
            
            history = model.fit(X_train,
                                y_train,
                                callbacks=[es],
                                epochs=200, 
                                batch_size=1024,
                                validation_split=0.2,
                                shuffle=True,
                                verbose=1)
            
            history_dict = history.history
            loss_values = history_dict['loss'] 
            val_loss_values = history_dict['val_loss'] 
            epochs = range(1, len(loss_values) + 1) 
            plt.plot(epochs, loss_values, 'bo', label='Training loss')
            plt.plot(epochs, val_loss_values, 'orange', label='Validation loss')
            plt.title('Training and validation loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig('./'+results_dir+'/loss_'+sensor_geom+'_0P'+str(threshold - int(threshold))[2:]+'thresh_run'+str(run_iter)+'.png')
            plt.close()
            acc = history.history['sparse_categorical_accuracy']
            val_acc = history.history['val_sparse_categorical_accuracy']
            epochs = range(1, len(acc) + 1)
            plt.plot(epochs, acc, 'bo', label='Training accuracy')
            plt.plot(epochs, val_acc, 'orange', label='Validation accuracy')
            plt.title('Training and validation accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
            #np.max(val_acc)
            plt.savefig('./'+results_dir+'/accuracy_'+sensor_geom+'_0P'+str(threshold - int(threshold))[2:]+'thresh_run'+str(run_iter)+'.png')
            plt.close()
            preds = model.predict(X_test) 
            predictionsFiles =np.argmax(preds, axis=1)
            pd.DataFrame(predictionsFiles).to_csv("./"+results_dir+"/predictionsFiles_"+sensor_geom+"_0P"+str(threshold - int(threshold))[2:]+"thresh_run"+str(run_iter)+".csv",header='predict', index=False)
            pd.DataFrame(y_test).to_csv("./"+results_dir+"/testResults_"+sensor_geom+"_0P"+str(threshold - int(threshold))[2:]+"thresh_run"+str(run_iter)+".csv",header='true', index=False)
            plt.hist(y_test, bins=30)
            plt.show()
            plt.close()
            score = model.evaluate(X_test, y_test, verbose=0)
            print("Test loss:", score[0])
            print("Test accuracy:", score[1])
            
            disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predictionsFiles)
            disp.figure_.suptitle("Multiclassifier Confusion Matrix")
            print(f"Confusion matrix:\n{disp.confusion_matrix}")
            plt.savefig('./'+results_dir+'/confusionMatrix_'+sensor_geom+'_0P'+str(threshold - int(threshold))[2:]+'_run'+str(run_iter)+'.png')
            plt.show()
            plt.close()
            model.save_weights('./'+models_dir+'/trained_model_'+sensor_geom+'_0P'+str(threshold - int(threshold))[2:]+'_run'+str(run_iter)+'.weights.h5')
            model.save('./'+models_dir+'/trained_model_'+sensor_geom+'_0P'+str(threshold - int(threshold))[2:]+'_run'+str(run_iter)+'.h5')
