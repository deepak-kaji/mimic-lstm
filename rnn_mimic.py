''' Recurrent Neural Network in Keras for use on the MIMIC-III '''

import gc
from time import time
import os
import math
import pickle

import numpy as np
import pandas as pd
from pad_sequences import PadSequences
from processing_utilities import PandasUtilities
from attention_function import attention_3d_block as Attention 

from keras import backend as K
from keras.models import Model, Input, load_model #model_from_json
from keras.layers import Masking, Flatten, Embedding, Dense, LSTM, TimeDistributed
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras import regularizers
from keras import optimizers

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, classification_report
from sklearn.metrics import recall_score, precision_score
from sklearn.model_selection import StratifiedKFold

ROOT = "./mimic_database/mapped_elements/"
FILE = "CHARTEVENTS_reduced_24_hour_blocks_plus_admissions_plus_patients_plus_scripts_plus_icds_plus_notes.csv"

######################################
## MAIN ###
######################################

def get_synth_sequence(n_timesteps=14):

  """

  Returns a single synthetic data sequence of dim (bs,ts,feats)

  Args:
  ----
    n_timesteps: int, number of timesteps to build model for

  Returns:
  -------
    X: npa, numpy array of features of shape (1,n_timesteps,2)
    y: npa, numpy array of labels of shape (1,n_timesteps,1) 

  """

  X = np.array([[np.random.rand() for _ in range(n_timesteps)],[np.random.rand() for _ in range(n_timesteps)]])
  X = X.reshape(1, n_timesteps, 2)
  y = np.array([0 if x.sum() < 0.5 else 1 for x in X[0]])
  y = y.reshape(1, n_timesteps, 1)
  return X, y

def wbc_crit(x):
  if (x > 12 or x < 4) and x != 0:
    return 1
  else:
    return 0

def temp_crit(x):
  if (x > 100.4 or x < 96.8) and x != 0:
    return 1
  else:
    return 0

def return_data(synth_data=False, balancer=True, target='MI', 
                return_cols=False, tt_split=0.7, val_percentage=0.8,
                cross_val=False, mask=False, dataframe=False,
                time_steps=14, split=True, pad=True):

  """

  Returns synthetic or real data depending on parameter

  Args:
  -----
      synth_data : synthetic data is False by default
      balance : whether or not to balance positive and negative time windows 
      target : desired target, supports MI, SEPSIS, VANCOMYCIN or a known lab, medication
      return_cols : return columns used for this RNN
      tt_split : fraction of dataset to use fro training, remaining is used for test
      cross_val : parameter that returns entire matrix unsplit and unbalanced for cross val purposes
      mask : 24 hour mask, default is False
      dataframe : returns dataframe rather than numpy ndarray
      time_steps : 14 by default, required for padding
      split : creates test train splits
      pad : by default is True, will pad to the time_step value
  Returns:
  -------
      Training and validation splits as well as the number of columns for use in RNN  

  """

  if synth_data:
    no_feature_cols = 2
    X_train = []
    y_train = []

    for i in range(10000):
      X, y = get_synth_sequence(n_timesteps=14)
      X_train.append(X)
      y_train.append(y)
    X_TRAIN = np.vstack(X_train)
    Y_TRAIN = np.vstack(y_train)

  else:
    df = pd.read_csv(ROOT + FILE)

    if target == 'MI':
      df[target] = ((df['troponin'] > 0.4) & (df['CKD'] == 0)).apply(lambda x: int(x))

    elif target == 'SEPSIS':
      df['hr_sepsis'] = df['heart rate'].apply(lambda x: 1 if x > 90 else 0)
      df['respiratory rate_sepsis'] = df['respiratory rate'].apply(lambda x: 1 if x>20 else 0)
      df['wbc_sepsis'] = df['WBCs'].apply(wbc_crit) 
      df['temperature f_sepsis'] = df['temperature (F)'].apply(temp_crit) 
      df['sepsis_points'] = (df['hr_sepsis'] + df['respiratory rate_sepsis'] 
                          + df['wbc_sepsis'] + df['temperature f_sepsis'])
      df[target] = ((df['sepsis_points'] >= 2) & (df['Infection'] == 1)).apply(lambda x: int(x))
      del df['hr_sepsis']
      del df['respiratory rate_sepsis']
      del df['wbc_sepsis']
      del df['temperature f_sepsis']
      del df['sepsis_points']
      del df['Infection']

    elif target == 'PE':
      df['blood_thinner'] = (df['heparin']  + df['enoxaparin'] + df['fondaparinux']).apply(lambda x: 1 if x >= 1 else 0)
      df[target] = (df['blood_thinner'] & df['ct_angio']) 
      del df['blood_thinner']


    elif target == 'VANCOMYCIN':
      df['VANCOMYCIN'] = df['vancomycin'].apply(lambda x: 1 if x > 0 else 0)   
      del df['vancomycin']
 
    print('target made')

    df = df.select_dtypes(exclude=['object'])

    if pad:
      pad_value=0
      df = PadSequences().pad(df, 1, time_steps, pad_value=pad_value)
      print('There are {0} rows in the df after padding'.format(len(df)))

    COLUMNS = list(df.columns)

    if target == 'MI':
      toss = ['ct_angio', 'troponin', 'troponin_std', 'troponin_min', 'troponin_max', 'Infection', 'CKD']
      COLUMNS = [i for i in COLUMNS if i not in toss]
    elif target == 'SEPSIS':
      toss = ['ct_angio', 'Infection', 'CKD']
      COLUMNS = [i for i in COLUMNS if i not in toss]
    elif target == 'PE':
      toss = ['ct_angio', 'heparin', 'heparin_std', 'heparin_min',
              'heparin_max', 'enoxaparin', 'enoxaparin_std',
              'enoxaparin_min', 'enoxaparin_max', 'fondaparinux',
              'fondaparinux_std', 'fondaparinux_min', 'fondaparinux_max',
              'Infection', 'CKD']
      COLUMNS = [i for i in COLUMNS if i not in toss]
    elif target == 'VANCOMYCIN':
      toss = ['ct_angio', 'Infection', 'CKD']
      COLUMNS = [i for i in COLUMNS if i not in toss]

    COLUMNS.remove(target)

    if 'HADM_ID' in COLUMNS:
      COLUMNS.remove('HADM_ID')
    if 'SUBJECT_ID' in COLUMNS:
      COLUMNS.remove('SUBJECT_ID')
    if 'YOB' in COLUMNS:
      COLUMNS.remove('YOB')
    if 'ADMITYEAR' in COLUMNS:
      COLUMNS.remove('ADMITYEAR')

    if return_cols:
      return COLUMNS

    if dataframe:
      return (df[COLUMNS+[target,"HADM_ID"]]) 

#    bool_df = (df[COLUMNS+[target]] == pad_value)
#    bool_matrix = bool_df.values
#    print('BOOL MATRIX SHAPE')
#    print(bool_matrix.shape)
#    bool_matrix = bool_matrix.reshape(int(bool_matrix.shape[0]/time_steps),time_steps,bool_matrix.shape[1])
#    print('BOOL MATRIX RESHAPED')
#    print(bool_matrix.shape)

    MATRIX = df[COLUMNS+[target]].values
    print('THE MATRIX SHAPE IS {0}'.format(MATRIX.shape))
    MATRIX = MATRIX.reshape(int(MATRIX.shape[0]/time_steps),time_steps,MATRIX.shape[1])
    print('THE MATRIX NEW SHAPE IS {0}'.format(MATRIX.shape))

    ## note we are creating a second order bool matirx
    bool_matrix = (~MATRIX.any(axis=2))
    MATRIX[bool_matrix] = np.nan
    MATRIX = PadSequences().ZScoreNormalize(MATRIX)
    ## restore 3D shape to boolmatrix for consistency
    bool_matrix = np.isnan(MATRIX)
    MATRIX[bool_matrix] = pad_value 
   
    permutation = np.random.permutation(MATRIX.shape[0])
    MATRIX = MATRIX[permutation]
    bool_matrix = bool_matrix[permutation]

    X_MATRIX = MATRIX[:,:,0:-1]
    Y_MATRIX = MATRIX[:,:,-1]
    
    x_bool_matrix = bool_matrix[:,:,0:-1]
    y_bool_matrix = bool_matrix[:,:,-1]

    X_TRAIN = X_MATRIX[0:int(tt_split*X_MATRIX.shape[0]),:,:]
    Y_TRAIN = Y_MATRIX[0:int(tt_split*Y_MATRIX.shape[0]),:]
    Y_TRAIN = Y_TRAIN.reshape(Y_TRAIN.shape[0], Y_TRAIN.shape[1], 1)

    X_VAL = X_MATRIX[int(tt_split*X_MATRIX.shape[0]):int(val_percentage*X_MATRIX.shape[0])]
    Y_VAL = Y_MATRIX[int(tt_split*Y_MATRIX.shape[0]):int(val_percentage*Y_MATRIX.shape[0])]
    Y_VAL = Y_VAL.reshape(Y_VAL.shape[0], Y_VAL.shape[1], 1)

    x_val_boolmat = x_bool_matrix[int(tt_split*x_bool_matrix.shape[0]):int(val_percentage*x_bool_matrix.shape[0])]
    y_val_boolmat = y_bool_matrix[int(tt_split*y_bool_matrix.shape[0]):int(val_percentage*y_bool_matrix.shape[0])]
    y_val_boolmat = y_val_boolmat.reshape(y_val_boolmat.shape[0],y_val_boolmat.shape[1],1)

    X_TEST = X_MATRIX[int(val_percentage*X_MATRIX.shape[0])::]
    Y_TEST = Y_MATRIX[int(val_percentage*X_MATRIX.shape[0])::]
    Y_TEST = Y_TEST.reshape(Y_TEST.shape[0], Y_TEST.shape[1], 1)

    x_test_boolmat = x_bool_matrix[int(val_percentage*x_bool_matrix.shape[0])::]
    y_test_boolmat = y_bool_matrix[int(val_percentage*y_bool_matrix.shape[0])::]
    y_test_boolmat = y_test_boolmat.reshape(y_test_boolmat.shape[0],y_test_boolmat.shape[1],1)

    # shouldn't be necessary but is a sanity check
    X_TEST[x_test_boolmat] = pad_value
    Y_TEST[y_test_boolmat] = pad_value

    if balancer:
      TRAIN = np.concatenate([X_TRAIN, Y_TRAIN], axis=2)
      print(np.where((TRAIN[:,:,-1] == 1).any(axis=1))[0])
      pos_ind = np.unique(np.where((TRAIN[:,:,-1] == 1).any(axis=1))[0])
      print(pos_ind)
      np.random.shuffle(pos_ind)
      neg_ind = np.unique(np.where(~(TRAIN[:,:,-1] == 1).any(axis=1))[0])
      print(neg_ind)
      np.random.shuffle(neg_ind)
      length = min(pos_ind.shape[0], neg_ind.shape[0])
      total_ind = np.hstack([pos_ind[0:length], neg_ind[0:length]])
      np.random.shuffle(total_ind)
      ind = total_ind
      if target == 'MI':
        ind = pos_ind
      else:
        ind = total_ind
      X_TRAIN = TRAIN[ind,:,0:-1]
      Y_TRAIN = TRAIN[ind,:,-1]
      Y_TRAIN = Y_TRAIN.reshape(Y_TRAIN.shape[0], Y_TRAIN.shape[1], 1)

  no_feature_cols = X_TRAIN.shape[2]

  if mask:
    print('MASK ACTIVATED')
    X_TRAIN = np.concatenate([np.zeros((X_TRAIN.shape[0], 1, X_TRAIN.shape[2])), X_TRAIN[:,1::,::]], axis=1)
    X_VAL = np.concatenate([np.zeros((X_VAL.shape[0], 1, X_VAL.shape[2])), X_VAL[:,1::,::]], axis=1)

  if cross_val:
    return (MATRIX, no_feature_cols)
    
  if split == True:
    return (X_TRAIN, X_VAL, Y_TRAIN, Y_VAL, no_feature_cols,
            X_TEST, Y_TEST, x_test_boolmat, y_test_boolmat,
            x_val_boolmat, y_val_boolmat)

  elif split == False:
    return (np.concatenate((X_TRAIN,X_VAL), axis=0),
            np.concatenate((Y_TRAIN,Y_VAL), axis=0), no_feature_cols) 

def build_model(benchmark=False, attention=False, no_feature_cols=None, time_steps=7, output_summary=False):

  """

  Assembles RNN with input from return_data function

  Args:
  ----
  no_feature_cols : The number of features being used AKA matrix rank
  time_steps : The number of days in a time block
  output_summary : Defaults to False on returning model summary

  Returns:
  ------- 
  Keras model object

  """
  if not benchmark and attention:
    print("time_steps:{0}|no_feature_cols:{1}".format(time_steps,no_feature_cols)) 
    input_layer = Input(shape=(time_steps, no_feature_cols)) 
    x = Attention(input_layer, time_steps)
    x = Masking(mask_value=0, input_shape=(time_steps, no_feature_cols))(x) 
    x = LSTM(256, return_sequences=True)(x)
    preds = TimeDistributed(Dense(1, activation="sigmoid"))(x)
    model = Model(inputs=input_layer, outputs=preds)
  
    RMS = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08)
    model.compile(optimizer=RMS, loss='binary_crossentropy', metrics=['acc'])
  
    if output_summary:
      model.summary()
    return model

  elif not benchmark and not attention:
    print("time_steps:{0}|no_feature_cols:{1}".format(time_steps,no_feature_cols)) 
    input_layer = Input(shape=(time_steps, no_feature_cols)) 
    x = Masking(mask_value=0, input_shape=(time_steps, no_feature_cols))(input_layer) 
    x = LSTM(256, return_sequences=True)(x)
    preds = TimeDistributed(Dense(1, activation="sigmoid"))(x)
    model = Model(inputs=input_layer, outputs=preds)
  
    RMS = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08)
    model.compile(optimizer=RMS, loss='binary_crossentropy', metrics=['acc'])
  
    if output_summary:
      model.summary()
    return model
     

  elif benchmark:
    print("time_steps:{0}|no_feature_cols:{1}".format(time_steps,no_feature_cols)) 
    input_layer = Input(shape=(time_steps, no_feature_cols)) 
    preds = TimeDistributed(Dense(1, activation="sigmoid"))(input_layer)
    model = Model(inputs=input_layer, outputs=preds)
  
    RMS = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=RMS, loss='binary_crossentropy', metrics=['acc'])
  
    if output_summary:
      model.summary()
    return model
    
def train(model_name="kaji_mach_0", synth_data=False, target='MI',
          benchmark=False, balancer=True, predict=False, return_model=False,
          n_percentage=1.0, time_steps=14, epochs=10, attention=False):

  """

  Use Keras model.fit using parameter inputs

  Args:
  ----
  model_name : Parameter used for naming the checkpoint_dir
  synth_data : Default to False. Allows you to use synthetic or real data.

  Return:
  -------
  Nonetype. Fits model only. 

  """

  f = open('./pickled_objects/X_TRAIN_{0}.txt'.format(target), 'rb')
  X_TRAIN = pickle.load(f)
  f.close()

  f = open('./pickled_objects/Y_TRAIN_{0}.txt'.format(target), 'rb')
  Y_TRAIN = pickle.load(f)
  f.close()
  
  f = open('./pickled_objects/X_VAL_{0}.txt'.format(target), 'rb')
  X_VAL = pickle.load(f)
  f.close()

  f = open('./pickled_objects/Y_VAL_{0}.txt'.format(target), 'rb')
  Y_VAL = pickle.load(f)
  f.close()

  f = open('./pickled_objects/x_boolmat_val_{0}.txt'.format(target), 'rb')
  X_BOOLMAT_VAL = pickle.load(f)
  f.close()

  f = open('./pickled_objects/y_boolmat_val_{0}.txt'.format(target), 'rb')
  Y_BOOLMAT_VAL = pickle.load(f)
  f.close()

  f = open('./pickled_objects/no_feature_cols_{0}.txt'.format(target), 'rb')
  no_feature_cols = pickle.load(f)
  f.close()

  X_TRAIN = X_TRAIN[0:int(n_percentage*X_TRAIN.shape[0])]
  Y_TRAIN = Y_TRAIN[0:int(n_percentage*Y_TRAIN.shape[0])]

  #build model
  model = build_model(no_feature_cols=no_feature_cols, output_summary=True, 
                      time_steps=time_steps, benchmark=benchmark, attention=attention)

  #init callbacks
  tb_callback = TensorBoard(log_dir='./logs/{0}_{1}.log'.format(model_name, time),
    histogram_freq=0,
    write_grads=False,
    write_images=True,
    write_graph=True) 

  #Make checkpoint dir and init checkpointer
  checkpoint_dir = "./saved_models/{0}".format(model_name)

  if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

  checkpointer = ModelCheckpoint(
    filepath=checkpoint_dir+"/model.{epoch:02d}-{val_loss:.2f}.hdf5",
    monitor='val_loss',
    verbose=0,
    save_best_only=True,
    save_weights_only=False,
    mode='auto',
    period=1)

  #fit
  model.fit(
    x=X_TRAIN,
    y=Y_TRAIN,
    batch_size=16,
    epochs=epochs,
    callbacks=[tb_callback], #, checkpointer],
    validation_data=(X_VAL, Y_VAL),
    shuffle=True)

  model.save('./saved_models/{0}.h5'.format(model_name))

  if predict:
    print('TARGET: {0}'.format(target))
    Y_PRED = model.predict(X_VAL)
    Y_PRED = Y_PRED[~Y_BOOLMAT_VAL]
    np.unique(Y_PRED)
    Y_VAL = Y_VAL[~Y_BOOLMAT_VAL]
    Y_PRED_TRAIN = model.predict(X_TRAIN)
    print('Confusion Matrix Validation')
    print(confusion_matrix(Y_VAL, np.around(Y_PRED)))
    print('Validation Accuracy')
    print(accuracy_score(Y_VAL, np.around(Y_PRED)))
    print('ROC AUC SCORE VAL')
    print(roc_auc_score(Y_VAL, Y_PRED))
    print('CLASSIFICATION REPORT VAL')
    print(classification_report(Y_VAL, np.around(Y_PRED)))

  if return_model:
    return model

def return_loaded_model(model_name="kaji_mach_0"):

  loaded_model = load_model("./saved_models/{0}.h5".format(model_name))

  return loaded_model

def cross_validation(target, time_steps, n_splits=5, benchmark=False):

  cvscores = []
  rocs = []
  sensitivities = []
  specificities = []

  (MATRIX, no_feature_cols) = return_data(time_steps=time_steps, target=target, cross_val=True)

  data_indices = np.array(range(MATRIX.shape[0]))
  bins = np.linspace(data_indices[0], data_indices[-1]+1, n_splits+1)
  digitized = np.digitize(data_indices, bins)
  list_of_arrays = [data_indices[digitized == i] for i in range(1, len(bins))]

  for i in range(n_splits):

    temp_list_of_arrays = [data_indices[digitized == i] for i in range(1, len(bins))] 
    del temp_list_of_arrays[i]
    train_array = np.hstack(temp_list_of_arrays)
    test_array = list_of_arrays[i]
 
    x_train = None
    x_test = None
    y_train = None
    y_test = None
    X_TRAIN = None
    Y_TRAIN = None

    x_train =  MATRIX[:,:,0:-1][train_array]
    x_test = MATRIX[:,:,0:-1][test_array]
    y_train = MATRIX[:,:,-1][train_array]
    y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], 1)
    y_test = MATRIX[:,:,-1][test_array]
    y_test = y_test.reshape(y_test.shape[0], y_test.shape[1], 1)

    TRAIN = np.concatenate([x_train, y_train], axis=2)
    pos_ind = np.unique(np.where(TRAIN[:,:,-1].sum(axis=1) != 0)[0])
    np.random.shuffle(pos_ind)
    neg_ind = np.unique(np.where(TRAIN[:,:,-1].sum(axis=1) == 0)[0])
    np.random.shuffle(neg_ind)
    length = min(pos_ind.shape[0], neg_ind.shape[0])
    total_ind = np.hstack([pos_ind[0:length], neg_ind[0:length]])
    special_ind = np.hstack([pos_ind[0:length], neg_ind[0:int(length)]])
    np.random.shuffle(total_ind)
    X_TRAIN = TRAIN[pos_ind,:,0:-1]
    Y_TRAIN = TRAIN[pos_ind,:,-1]
    Y_TRAIN = Y_TRAIN.reshape(Y_TRAIN.shape[0], Y_TRAIN.shape[1], 1)

#    if mask:
#      print('MASK ACTIVATED')
#      X_TRAIN = np.concatenate([np.zeros((X_TRAIN.shape[0], mask, X_TRAIN.shape[2])), X_TRAIN[:,mask::,::]], axis=1)
#      x_test = np.concatenate([np.zeros((x_test.shape[0], mask, x_test.shape[2])), x_test[:,mask::,::]], axis=1)

    model = None
    model = build_model(no_feature_cols=no_feature_cols,
                        time_steps=time_steps,
                        output_summary=False,
                        benchmark=benchmark)

    model.fit(X_TRAIN,
              Y_TRAIN,
              batch_size=32,
              validation_data=(x_test, y_test),
              epochs=14,
              verbose=1)

    Y_PRED = None
    Y_PRED = model.predict(x_test)
    print('\n')
    accuracy = None
    accuracy = accuracy_score(y_test.flatten(), np.around(Y_PRED.flatten()))
    roc = roc_auc_score(y_test.flatten(), Y_PRED.flatten())
    specificity = precision_score(y_test.flatten(), np.around(Y_PRED.flatten()))
    sensitivity = recall_score(y_test.flatten(), np.around(Y_PRED.flatten()))
    print('Accuracy Scores: {0}'.format(accuracy)) 
    print('AUC ROC: {0}'.format(roc))
    print('Specificity: {0}'.format(specificity))
    print('Sensitivity: {0}'.format(sensitivity))
    print(confusion_matrix(y_test.flatten(), np.around(Y_PRED.flatten())))
    cvscores.append(accuracy) 
    rocs.append(roc)
    sensitivities.append(sensitivity)
    specificities.append(specificity)

  print('\n')
  print(target)
  print('Accuracy Score')
  print('{0}% (+/- {1}%)'.format(np.mean(cvscores), np.std(cvscores)))  
  print('AUC')
  print('{0}% (+/- {1}%)'.format(np.mean(rocs), np.std(rocs)))  
  print('Sensitivity Score')
  print('{0}% (+/- {1}%)'.format(np.mean(sensitivities), np.std(sensitivities)))  
  print('Specificites Score')
  print('{0}% (+/- {1}%)'.format(np.mean(specificities), np.std(specificities)))  

def pickle_objects(target='MI', time_steps=14):

  (X_TRAIN, X_VAL, Y_TRAIN, Y_VAL, no_feature_cols,
   X_TEST, Y_TEST, x_boolmat_test, y_boolmat_test,
   x_boolmat_val, y_boolmat_val) = return_data(balancer=True, target=target,
                                                            pad=True,
                                                            split=True, 
                                                      time_steps=time_steps)  

  features = return_data(return_cols=True, synth_data=False,
                         target=target, pad=True, split=True,
                         time_steps=time_steps)

  f = open('./pickled_objects/X_TRAIN_{0}.txt'.format(target), 'wb')
  pickle.dump(X_TRAIN, f)
  f.close()

  f = open('./pickled_objects/X_VAL_{0}.txt'.format(target), 'wb')
  pickle.dump(X_VAL, f)
  f.close()

  f = open('./pickled_objects/Y_TRAIN_{0}.txt'.format(target), 'wb')
  pickle.dump(Y_TRAIN, f)
  f.close()

  f = open('./pickled_objects/Y_VAL_{0}.txt'.format(target), 'wb')
  pickle.dump(Y_VAL, f)
  f.close()

  f = open('./pickled_objects/X_TEST_{0}.txt'.format(target), 'wb')
  pickle.dump(X_TEST, f)
  f.close()

  f = open('./pickled_objects/Y_TEST_{0}.txt'.format(target), 'wb')
  pickle.dump(Y_TEST, f)
  f.close()

  f = open('./pickled_objects/x_boolmat_test_{0}.txt'.format(target), 'wb')
  pickle.dump(x_boolmat_test, f)
  f.close()

  f = open('./pickled_objects/y_boolmat_test_{0}.txt'.format(target), 'wb')
  pickle.dump(y_boolmat_test, f)
  f.close()

  f = open('./pickled_objects/x_boolmat_val_{0}.txt'.format(target), 'wb')
  pickle.dump(x_boolmat_val, f)
  f.close()

  f = open('./pickled_objects/y_boolmat_val_{0}.txt'.format(target), 'wb')
  pickle.dump(y_boolmat_val, f)
  f.close()

  f = open('./pickled_objects/no_feature_cols_{0}.txt'.format(target), 'wb')
  pickle.dump(no_feature_cols, f)
  f.close()

  f = open('./pickled_objects/features_{0}.txt'.format(target), 'wb')
  pickle.dump(features, f)
  f.close()


if __name__ == "__main__":

#    pickle_objects(target='MI', time_steps=14)#
#    K.clear_session()
#    pickle_objects(target='SEPSIS', time_steps=14)
#    K.clear_session()
#    pickle_objects(target='VANCOMYCIN', time_steps=14)
#
##  cross_validation('MI', 14, n_splits=5) #, benchmark=True) #, mask=8)
##  cross_validation('SEPSIS', 14, n_splits=5) # benchmark=True)
##  cross_validation('VANCOMYCIN', 14, n_splits=5) #benchmark=True)
#
### BIG THREE ##
#
#    K.clear_session()
#    train(model_name='kaji_mach_final_no_mask_MI_pad14', epochs=13,
#          synth_data=False, predict=True, target='MI', time_steps=14)
#
#    K.clear_session()
#
#    train(model_name='kaji_mach_final_no_mask_VANCOMYCIN_pad14', epochs=14,
#          synth_data=False, predict=True, target='VANCOMYCIN', time_steps=14) 
#
#    K.clear_session()
#
#    train(model_name='kaji_mach_final_no_mask_SEPSIS_pad14', epochs=17,
#          synth_data=False, predict=True, target='SEPSIS', time_steps=14) 
#
####
#
#    K.clear_session()
#    train(model_name='kaji_mach_final_no_mask_MI_pad14_attention', epochs=13,
#          synth_data=False, predict=True, target='MI', time_steps=14,
#          attention=True)
#
#    K.clear_session()
#
#    train(model_name='kaji_mach_final_no_mask_VANCOMYCIN_pad14_attention', epochs=14,
#          synth_data=False, predict=True, target='VANCOMYCIN', time_steps=14,
#          attention=True) 
#
#    K.clear_session()
#
#    train(model_name='kaji_mach_final_no_mask_SEPSIS_pad14_attention', epochs=17,
#          synth_data=False, predict=True, target='SEPSIS', time_steps=14,
#          attention=True) 
#
## BIG THREE BENCHMARK ##

    train(model_name='kaji_mach_final_no_mask_MI_pad14_bench', 
          epochs=13, synth_data=False, predict=True, target='MI',
          time_steps=14, benchmark=True) #, mask=4)
  
    train(model_name='kaji_mach_final_no_mask_VANCOMYCIN_pad14_bench',
          epochs=14, synth_data=False, predict=True, target='VANCOMYCIN',
          time_steps=14, benchmark=True) 
  
    train(model_name='kaji_mach_final_no_mask_SEPSIS_pad14_bench',
          epochs=17, synth_data=False, predict=True, target='SEPSIS',
          time_steps=14, benchmark=True) 

## REDUCE SAMPLE SIZES ##

## MI ##

    train(model_name='kaji_mach_final_no_mask_MI_pad14_80_percent', epochs=13,
          synth_data=False, predict=True, target='MI', time_steps=14,
          n_percentage=0.80)
  
    K.clear_session()
  
    train(model_name='kaji_mach_final_no_mask_MI_pad14_60_percent', epochs=13,
          synth_data=False, predict=True, target='MI', time_steps=14,
          n_percentage=0.60)
  
    K.clear_session()
  
    train(model_name='kaji_mach_final_no_mask_MI_pad14_40_percent', epochs=13,
          synth_data=False, predict=True, target='MI', time_steps=14,
          n_percentage=0.40)
  
    K.clear_session()
  
    train(model_name='kaji_mach_final_no_mask_MI_pad14_20_percent', epochs=13,
          synth_data=False, predict=True, target='MI', time_steps=14,
          n_percentage=0.20)
  
    K.clear_session()
  
    train(model_name='kaji_mach_final_no_mask_MI_pad14_10_percent', epochs=13,
          synth_data=False, predict=True, target='MI', time_steps=14,
          n_percentage=0.10)
  
    K.clear_session()
  
    train(model_name='kaji_mach_final_no_mask_MI_pad14_5_percent', epochs=13,
          synth_data=False, predict=True, target='MI', time_steps=14,
          n_percentage=0.05)
  
    K.clear_session()
  
# SEPSIS ##
 
    train(model_name='kaji_mach_final_no_mask_VANCOMYCIN_pad14_80_percent',
          epochs=14,synth_data=False, predict=True, target='VANCOMYCIN',
          time_steps=14, n_percentage=0.80) 
  
    K.clear_session()
  
    train(model_name='kaji_mach_final_no_mask_VANCOMYCIN_pad14_60_percent',
          epochs=14, synth_data=False, predict=True, target='VANCOMYCIN',
          time_steps=14, n_percentage=0.60) 
  
    K.clear_session()
  
    train(model_name='kaji_mach_final_no_mask_VANCOMYCIN_pad14_40_percent',
          epochs=14, synth_data=False, predict=True, target='VANCOMYCIN',
          time_steps=14, n_percentage=0.40) 
  
    K.clear_session()
  
    train(model_name='kaji_mach_final_no_mask_VANCOMYCIN_pad14_20_percent', epochs=14,
          synth_data=False, predict=True, target='VANCOMYCIN', time_steps=14,
          n_percentage=0.20) 
  
    K.clear_session()
  
    train(model_name='kaji_mach_final_no_mask_VANCOMYCIN_pad14_10_percent',
          epochs=13, synth_data=False, predict=True, target='VANCOMYCIN',
          time_steps=14, n_percentage=0.10)
  
    K.clear_session()
  
    train(model_name='kaji_mach_final_no_mask_VANCOMYCIN_pad14_5_percent',
          epochs=13, synth_data=False, predict=True, target='VANCOMYCIN',
          time_steps=14, n_percentage=0.05)
 
# VANCOMYCIN ##
 
    train(model_name='kaji_mach_final_no_mask_SEPSIS_pad14_80_percent',
          epochs=17, synth_data=False, predict=True, target='SEPSIS',
          time_steps=14, n_percentage=0.80) 
  
    K.clear_session()
  
    train(model_name='kaji_mach_final_no_mask_SEPSIS_pad14_60_percent',
          epochs=17, synth_data=False, predict=True, target='SEPSIS',
          time_steps=14, n_percentage=0.60) 
  
    K.clear_session()
  
    train(model_name='kaji_mach_final_no_mask_SEPSIS_pad14_40_percent',
          epochs=17, synth_data=False, predict=True, target='SEPSIS',
          time_steps=14, n_percentage=0.40) 
  
    K.clear_session()
  
    train(model_name='kaji_mach_final_no_mask_SEPSIS_pad14_20_percent',
          epochs=17, synth_data=False, predict=True, target='SEPSIS',
          time_steps=14, n_percentage=0.20) 
  
    K.clear_session()
  
    train(model_name='kaji_mach_final_no_mask_SEPSIS_pad14_10_percent',
          epochs=13, synth_data=False, predict=True, target='SEPSIS',
          time_steps=14, n_percentage=0.10)
  
    K.clear_session()
  
    train(model_name='kaji_mach_final_no_mask_SEPSIS_pad14_5_percent',
          epochs=13, synth_data=False, predict=True, target='SEPSIS',
          time_steps=14, n_percentage=0.05)

