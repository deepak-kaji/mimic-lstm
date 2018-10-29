''' Meant to deal with padding for an RNN when keras.preprocessing.pad_sequences fails '''

import numpy as np
import pandas as pd

class PadSequences(object):
    
    def __init__(self):
        self.name = 'padder'

    def pad(self, df, lb, time_steps, pad_value=-100):

        ''' Takes a file path for the dataframe to operate on. lb is a lower bound to discard 
            ub is an upper bound to truncate on. All entries are padded to their ubber bound '''

        #df = pd.read_csv(path):
        self.uniques = pd.unique(df['HADM_ID'])
        df = df.groupby('HADM_ID').filter(lambda group: len(group) > lb).reset_index(drop=True)
        df = df.groupby('HADM_ID').apply(lambda group: group[0:time_steps]).reset_index(drop=True)
        df = df.groupby('HADM_ID').apply(lambda group: pd.concat([group, pd.DataFrame(pad_value*np.ones((time_steps-len(group), len(df.columns))), columns=df.columns)], axis=0)).reset_index(drop=True)
         
        return df

    def ZScoreNormalize(self, matrix):

        ''' Performs Z Score Normalization for 3rd order tensors 
            matrix should be (batchsize, time_steps, features) 
            Padded time steps should be masked with np.nan '''

        x_matrix = matrix[:,:,0:-1]
        y_matrix = matrix[:,:,-1]
        print(y_matrix.shape)
        y_matrix = y_matrix.reshape(y_matrix.shape[0],y_matrix.shape[1],1)
        means = np.nanmean(x_matrix, axis=(0,1))
        stds = np.nanstd(x_matrix, axis=(0,1))
        print(x_matrix.shape)
        print(means.shape)
        print(stds.shape)
        x_matrix = x_matrix-means
        print(x_matrix.shape)
        x_matrix = x_matrix / stds
        print(x_matrix.shape)
        print(y_matrix.shape)
        matrix = np.concatenate([x_matrix, y_matrix], axis=2)
        
        return matrix
    
    def MinMaxScaler(self, matrix, pad_value=-100):

        ''' Performs a NaN/pad-value insensiive MinMaxScaling 
            When column maxs are zero, it ignores these columns for division '''

        bool_matrix = (matrix == pad_value)
        matrix[bool_matrix] = np.nan
        mins = np.nanmin(matrix, axis=0)
        maxs = np.nanmax(matrix, axis=0)
        matrix = np.divide(np.subtract(matrix, mins), np.subtract(maxs,mins), where=(np.nanmax(matrix,axis=0) != 0))
        matrix[bool_matrix] = pad_value

        return matrix
        








