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
        from sklearn.cluster import KMeans
        
        #cluster the operational settings into 6
        if development_mode:
            self.kmeans = KMeans(n_clusters=6,random_state=0).fit(df[self.op_settings])
            df['op_mode'] = self.kmeans.labels_
        else:
            df['op_mode'] = self.kmeans.predict(df[self.op_settings])
            
        return df
    
    def train_test_split(self, df):
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
        self.train_max = trainset.groupby(["op_mode"])[self.sensor_name].max().reset_index()                                                                                     
        self.train_min = trainset.groupby(["op_mode"])[self.sensor_name].min().reset_index()                                                                           
        
        return trainset, evalset
    
    def minmax_scale(self, df):        
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

        """TODO: for now, it is harcoded for only one column
        I need to find a more efficient way to change this for all the columns.
        this is beause of df.assign(), where "s12=(...)" 
        cannot be passed as string or an array"""   
        scaled_df = pd.DataFrame()    
        scaled_df = df.assign(s12=(
            df.s12 - mins)/(maxes - mins))   
        return scaled_df

    
    def denoise_sensors(self, df):
        denoised_df = pd.DataFrame(columns=df.columns)
        for col in df.columns:
            if col.startswith("s"):
                denoised_sensor=[]
                for u in df.unit.unique():
                    # 5 is the alpha for denoising the sensor
                    # the higher the alpha, the smoother the signal
                    denoised_sensor += list(ndimage.gaussian_filter1d(df[col][df.unit==u], 8))
                    
                df[col]=denoised_sensor
            else:
                df[col] = df[col].values
                
                
        scaled_df = df.copy()
        return df
    
    def calculate_TTF(self, df):
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
        """for this step, we need to have the TTF already calculated"""
        
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
        df["nRUL"] = df.groupby(['unit'])["RUL"].transform(
            lambda x: (x-min(x))/(max(x)-min(x))
        ).values
        
        df["health"]=[1 if r!=1 else 0 
                                  for r in df["nRUL"].values]
    
        del df["nRUL"]
        
        print("Succesfully calculated the Heath States.")
        return df

    
    def visualize_healthstatus(self, df, n_units):
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