import sys

import pandas as pd
import sklearn as sk
import scipy.ndimage as ndimage
from scipy.fft import fft, fftfreq
import numpy as np
np.random.seed(7)

#visualizations
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
sns.set_style("whitegrid")

class CMAPSS():
    
    def __init__(self):
        self.sensor_names = list()
        self.sensor_name = ""
        self.op_settings = list()
        
        self.train_min = None
        self.train_max = None

        self.kmeans = None
        
    
    def load_data(self, path):
        """
        This function reads the .txt file from the given path and returns the CMAPSS dataframe.
        the operation setting columns, as well as the sensors (the ones with no nan values)
        are globally saved.
        
        Input: path as a string
        Output: a dataframe
        """
        try:      
            df = pd.read_csv(path, sep=' ', header=None)
            
        except Exception as e:
            print("Cannot read the file: Incorrect path.\n{}".format(e))
            sys.exit()
            
        try: 
            df = df.rename(index=str,
                                     columns={0: "unit", 1: "cycle",
                                              2: "os_1", 3: "os_2",
                                              4: "os_3"})

            self.op_settings = ['os_1', 'os_2', 'os_3']

            for col in df.columns:
                if isinstance(col, str): 
                    continue

                df = df.rename(index=str,
                                         columns={col: 's' + str(col-4)})

            self.sensor_names = ['s'+str(i+1) for i in range(20)]

        except Exception:
            print("Incorrect shape: expected 28 dimentions.")
            print("Please make sure your data is the NASTA PHM08 data.")
            sys.exit()
    
        print("Succesfully loaded PHM08 data from {}".format(path))
        return df.dropna(axis=1,how='all')

    
    def cluster_operational_settings(self, df, development_mode):
        """
        This function clusters the development dataset based on its six different operational modes.
        The labels of the clusters are then added as the column "op_mode", in which each value represents a
        different cluster.
        These 6 clusters were visualized during the exploratory analysis and confirmed by the 
        published works on this dataset. The boolean parameter "development_mode" determines if the given 
        dataframe is the development or test dataset. If it is the test dataset, then the fitted 
        kmeans will only predict the labels of the data points.
        
        Input: the development dataframe and a boolean variable
        Output: a dataframe with one extra column "op_mode" 
        """
        
        
        from sklearn.cluster import KMeans
        
        #cluster the operational settings into 6
        if development_mode:
            self.kmeans = KMeans(n_clusters=6,random_state=0).fit(df[self.op_settings])
            df['op_mode'] = self.kmeans.labels_
        else:
            df['op_mode'] = self.kmeans.predict(df[self.op_settings])
            
        return df
    
    def train_test_split(self, df):
        """
        This function splits the train and the test set to create unseen data.
        Please note that CMAPSS data already provides us with a test.txt file. however, this test data does 
        not have labels, and therefore, I did not use them for my final evaluation.
        Moreover, we save the min and max of each operation mode of each engine from the trainset.
        This ensures no information leaks to the test set, and the test set is
        normalized with the statistics that the model has already learned.
        
        Input: a dataframe
        Output: the train and test dataframe
        """
   

        from sklearn.model_selection import GroupShuffleSplit
        
        trainset, evalset = pd.DataFrame(), pd.DataFrame()
        groups = df.unit.values
        shufflesplit = GroupShuffleSplit(n_splits=1, train_size=.7, random_state=7)
        
        for train_idx, test_idx in shufflesplit.split(df, df.health.values, groups):  
            trainset = df.iloc[train_idx].copy().reset_index(drop=True)
            trainset["op_mode"] = df.iloc[train_idx].op_mode.values
            trainset["health"] = df.iloc[train_idx].health.values

            evalset = df.iloc[test_idx].copy().reset_index(drop=True)
            evalset["op_mode"] = df.iloc[test_idx].op_mode.values
            evalset["health"] = df.iloc[test_idx].health.values
        
        # get the trainset statistics for the normalization step
        self.train_max = trainset.groupby(["unit","op_mode"])[self.sensor_name].max().reset_index()                                                                                     
        self.train_min = trainset.groupby(["unit","op_mode"])[self.sensor_name].min().reset_index()                                                                           
        
        return trainset, evalset
    
    def minmax_scale(self, df):   
        """
        This function, min-max-normalizes the input dataframe. the min and max values are the global train min 
        and max values, saved from the train and test split function.
        Since the size of the dataframe can vary, we need to repeat min and max for each operation mode and 
        create a sorted list, which has the same size as the input dataframe. 
        for example: 
        size(df)=20 and df contains the timeseries  X: [x1, x2, x3, ..., x20]
        and column "op_mode:[1,3,2,0,0,1,3,2,2,0,0,1,1,3,2,0,0,1,1,1]
        
        sorted_op_mode= [0,0,0,0,0,0,1,1,1,1,1,1,1,2,2,2,2,3,3,3]: size()=20
        operation_mod_train_max = [max0, max0, max0, ... max3, max3, max3]; size()=20  
        operation_mod_train_min = [min0, min0, min0, ... min3, min3, min3]; size()=20  
        We then sort the min and maxes based on their indices to ensure they correspond to 
        the correct data points.  
        
        Input: a dataframe
        Output: min-max normalized dataframe
        """
        # now create a list of sensor max and mins for each operation mode 
        # which has the size of your df
        sorted_maxes = []
        sorted_mins = []
        operation_modes = list(self.train_max.op_mode.unique())

        for om in operation_modes:
            op_mode_max = self.train_max[self.train_max.op_mode == om][self.sensor_name].values[0]
            sorted_maxes += [op_mode_max for _ in range(len(df[df.op_mode==om]))]

            op_mode_min = self.train_min[self.train_min.op_mode == om][self.sensor_name].values[0]
            sorted_mins += [op_mode_min for _ in range(len(df[df.op_mode==om]))]

        stat_df = df.sort_values("op_mode")
        stat_df["max"] = sorted_maxes
        stat_df["min"] = sorted_mins
        stat_df = stat_df.sort_index()
        mins = stat_df["min"].values
        maxes = stat_df["max"].values

        scaled_df = pd.DataFrame()    
        scaled_df = df.assign(s12=(
            df.s12 - mins)/(maxes - mins))   
        return scaled_df

    
    def denoise_sensors(self, df):
        """
        This function, uses the gaussian filter to remove the noise from each normalized engine time series
        
        Input: a dataframe
        Output: denoised dataframe
        """
        denoised_df = pd.DataFrame(columns=df.columns)
        for col in df.columns:
            if col.startswith("s"):
                denoised_sensor=[]
                for u in df.unit.unique():
                    # 5 is the alpha for denoising the sensor
                    # the higher the alpha, the smoother the signal
                    denoised_sensor += list(ndimage.gaussian_filter1d(df[col][df.unit==u], 10))
                    
                df[col]=denoised_sensor
            else:
                df[col] = df[col].values
                
                
        scaled_df = df.copy()
        return df
    
    def calculate_TTF(self, df):
        """
        This function, this function calculates the Time to Failure (or, for this dataset, Cycle to Failure) 
        of each engine. We calculate this by reversing the number of executed thermodynamical cycles 
        from their maximum down to when the engine broke down. 
        Based on the CMAPSS documentation, the first cycle is always in a healthy state, and then its health
        starts to decrease until it goes to a breakdown point gradually.
        
        Input: a dataframe
        Output: a dataframe with the calculated TTF
        """
        # calculate number of cycles to failure
        df = pd.merge(df,
                           df.groupby('unit',
                                           as_index=False)['cycle'].max(),
                           how='left', on='unit')
        
        df.rename(columns={"cycle_x": "cycle", "cycle_y": "maxcycle"},
                       inplace=True)
        df['TTF'] = df['maxcycle'] - df['cycle']
        df['TTF'] = df['TTF'].values
        print("Succesfully calculated the CTFs (Cycles to Failure).")
        
        return df
    
    
    def calculate_continues_healthstate(self, df):
        """
        This function calculates the Remaining Useful Life (RUL), proposed by the winner of the challenge. 
        They used a piecewise linear function to calculate the healthy/stable duration of the engine and 
        then the unhealthy duration of the engine. The maximum life is a fixed value, also proposed by the 
        winner of the challenge. IMPORTANT: We need to have the TTF already calculated for this step.
        
        Input: a dataframe
        Output: a dataframe with the calculated RUL   
        """
        
        # taken from the winner of the challenge
        max_life = 120

        # calculate the RULs using a piecewise linear function
        RULs = []
        for unit in df.unit.unique():
            max_cycle = int(df[df.unit==unit].TTF.max())
            knee_point = max_cycle - max_life
            stable_life = max_life
            
            if knee_point<1:
                RULs += list(df[df.unit==unit].TTF)
                continue
                
            else:
                kink_RUL = []
                for i in range(len(df[df.unit==unit])):
                    if i < knee_point:
                        kink_RUL.append(max_life)
                    else:
                        tmp = kink_RUL[i-1]-(stable_life/(max_cycle-knee_point))
                        if tmp<0:tmp=0
                        kink_RUL.append(np.round(tmp, 3))

                RULs += kink_RUL

        df['RUL'] = RULs
        print("Succesfully calculated the RULs (Remaining Useful Lives).")
        
        return df
    
    
    def calculate_descrete_healthstate(self, df):
        """
        This function labels the health state of the engine by looking at the Remaining Useful Life values. 
        I transformed this regression task into a binary classification task to predict the health state. 
        (healthy vs. unhealthy) of the engines. I assume that when the RUL is stable, the machine is healthy, 
        and when RUL reaches the knee point and degrades, I annotate the data points as unhealthy. 
        
        Input: a dataframe
        Output: a dataframe with calculated binary labels
        """
        
        df["nRUL"] = df.groupby(['unit'])["RUL"].transform(
            lambda x: (x-min(x))/(max(x)-min(x))
        ).values
        
        df["health"]=[1 if r!=1 else 0 
                                  for r in df["nRUL"].values]
    
        del df["nRUL"]
        
        print("Succesfully calculated the Heath States.")
        return df

    
    def visualize_healthstatus(self, df, n_units):
        """
        This function visualized all three labels: Time to Failure, Remaining Useful Life and 
        the Health state.
        
        Input: a dataframe, given number of engines to visualize
        Output: a plot of all three labels for n units/engines
        """
        
        
        if n_units>15:
            msg="{} engines to visualize is too large!".format(n_units)
            msg+=" Please choose a number between 1 and 15"
            return msg
            
        n_samples = df[(df.unit==n_units)&(df.RUL==0)].index[0]
        x_axis_size=10; y_axis_size=3
        ax_font=12; title_font=16
        if n_units>7: 
            x_axis_size=n_units*2; y_axis_size*=2
            ax_font=16; title_font=20
            
        plt.figure(figsize=(x_axis_size, y_axis_size));
        plt.title("Health Status of {} Engines".format(n_units), fontsize=title_font)
        plt.xticks(fontsize=ax_font)
        plt.yticks(fontsize=ax_font)
        df["RUL"][:n_samples].plot(c="green", alpha=0.5,
                               linewidth=10, label="RUL")
        df["TTF"][:n_samples].plot(linestyle='--', c="black",
                               label="TTF")

        # now add the background colors based on the labeled health state
        for u in range(1,n_units+1):
            try:
                # heathy
                x_min = df[(df.unit==u) & (df.RUL==120)].index[0]
                x_max = df[(df.unit==u) & (df.RUL==120)].index[-1]
                
                plt.axvspan(xmin=x_min,
                            xmax=x_max,
                            ymax=1, 
                            facecolor='green', alpha=0.2)
            except Exception as e:
                # there is no healthy label available
                pass

            # unhealthy
            x_min = df[(df.unit==u) & (df.health==1)].index[0]
            x_max = df[(df.unit==u) & (df.health==1)].index[-1]  

            plt.axvspan(xmin=x_min,
                        xmax=x_max,
                        ymax=1,
                        facecolor='red', alpha=0.2)

        plt.legend(fontsize=ax_font)
        pass
    
    def visualize_denoised_sensors(self, df, n_units):
        """
        This function visualizes the normalized and denoised time series.
        I also added the Healthy state and the unhealthy state into the background as green and red colors.
        
        Input: a dataframe, given number of engines to visualize
        Output: a plot of denoised time series over the binary health state
        """
        
        
        if n_units>15:
            msg="{} engines to visualize is too large!".format(n_units)
            msg+=" Please choose a number between 1 and 15"
            return msg

        units = df.unit.unique()[:n_units]
        n_samples = df[
            (df.unit==units[-1])&
            (df.RUL==0)].index[0]
        col = [c for c in df.columns if c.startswith("s")]
        df[col][:n_samples].plot(linewidth=1, linestyle='-', c="black")
        plt.scatter(range(n_samples), df[col][:n_samples], c="black")
        plt.ylim(0,1,0.1)
        for u in units:
            # heathy
            try:
                x_min = df[(df.unit==u) &
                                       (df.RUL==120)].index[0]
                x_max = df[(df.unit==u) &
                                       (df.RUL==120)].index[-1]
                
                plt.axvspan(xmin=x_min,
                            xmax=x_max,
                            ymax=1, 
                            facecolor='green', alpha=0.2)
            except Exception as e:
                # there is no healthy label available
                pass



            # unhealthy
            try:
                x_min = df[(df.unit==u) &
                                       (df.RUL==120)].index[-1]+1
                x_max = df[(df.unit==u) &
                                       (df.RUL==0)].index[0]
            except Exception as e:
                x_min = df[(df.unit==u) & (df.health==1)].index[0]
                x_max = df[(df.unit==u) & (df.health==1)].index[-1]                

            plt.axvspan(xmin=x_min,
                        xmax=x_max,
                        ymax=1,
                        facecolor='red', alpha=0.2)
        pass


    def get_univariate_cmapss(self, df, s):
        """
        This function receives a dataframe and a column name, removes the other sensor time series data and 
        keep the input column as the only time series.
        Therefore, we transform the dataframe, from a multivariate time series into a univariate time series.
        
        Input: a dataframe and a column name (has to be a preprocessed sensor data)
        Output: a dataframe with only the selected sensor time series data
        """
        
        
        selected_cols = [col for col in df.columns
                         if not col.startswith("s")] + [s]
        self.sensor_name = s
        return df[selected_cols]
    
    
    def rounder(self, x):
        if (x-int(x) >= 0.5):
            return np.ceil(x)
        else:
            return np.floor(x)

    def window_data(self, df, window_size, hopsize):
        """
        This function is a rolling window over the timeseries, which chuncks the timeseries into smaller 
        sequences with the size "window_size". Each sequence has an overlap of size "hopsize" with its
        previous sequence.
        
        Input: a dataframe, an integer value for window size, and an integer value for the overlap size
        Output: an array of size (n, window_size)
        """
        
        
        data = []
        label = []
        for unit in df.unit.unique():
            engine = df[df.unit==unit].copy()
            
            if len(engine) < window_size:
                padding_size = window_size-len(engine)
                padded_values = [engine.s12.values[-1] for _ in range(padding_size)]
                data.append(engine.s12.values.tolist()+padded_values)
                label.append(int(self.rounder(engine.health.values.mean())))
                continue

            start=0
            while start+window_size < len(engine):            
                data.append(engine.s12[start:start+window_size].values)
                label.append(int(self.rounder(engine.health[start:start+window_size
                                                           ].values.mean())))
                start += hopsize

            data.append(engine.s12[-window_size:].values)
            label.append(int(self.rounder(engine.health[-window_size:].values.mean())))

        return np.array(data, dtype=object), label