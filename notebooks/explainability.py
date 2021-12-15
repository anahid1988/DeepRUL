import inspect_results as inspect
import prepare_data
from scipy.fft import fft, fftfreq
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
np.random.seed(7)

cmapss = prepare_data.CMAPSS()

def get_predictions_as_df(nd_array, y_true, y_pred):
    """
    This function 

    Input: a numpy array
    Output: a numpy array  
    """    
    
    
    df = pd.DataFrame(data=nd_array)
    df.columns = ["ts_{}".format(i) for i in range(nd_array.shape[1])] 
    df["y_true"] = y_true
    df["y_pred"] = inspect.continues_to_binary(y_pred)
    df["confidence"] = y_pred
    
    return df
    

def get_counter_and_factuals(train_cluster, train_cluster_df, selected_sample):
    """
    This function 

    Input: a numpy array
    Output: a numpy array  
    """    
    
    # get the x_test[i] and exclude y_true, y_pred and model confidence
    x_sample=selected_sample[["ts_{}".format(i) for i in range(len(selected_sample)-3)]].values

    x_prediction = selected_sample["y_pred"]
    x_actual_label = selected_sample["y_true"]
    x_model_confidence = selected_sample["confidence"]

    # to which healthy cluster from the trainset is this sample the closest?
    closest_healthy_cluster = train_cluster["healthy_clusters"
                                           ]["kmeans"
                                            ].predict(x_sample[np.newaxis, :])[0]

    closest_unhealthy_cluster = train_cluster["unhealthy_clusters"
                                           ]["kmeans"
                                            ].predict(x_sample[np.newaxis, :])[0]

    center_cols = ["center_{}".format(i) for i in range(len(x_sample))]
    
    factual_cluster = None
    counterfactual_cluster = None

    if int(x_prediction) == 1:
        # assuming your factual cluster is an unhealthy cluster
        factual_cluster = train_cluster_df[
            train_cluster_df.cluster==closest_unhealthy_cluster
        ]

        # and your counterfactual cluster is a healthy cluster
        counterfactual_cluster = train_cluster_df[
                train_cluster_df.cluster==closest_healthy_cluster
            ]

    # but if the prediction for this sample was healthy
    elif int(x_prediction) == 0:
        # then your factual cluster is a healthy cluster
        factual_cluster = train_cluster_df[
            train_cluster_df.cluster==closest_healthy_cluster
        ]

        # and your counterfactual cluster is an unhealthy cluster
        counterfactual_cluster = train_cluster_df[
            train_cluster_df.cluster==closest_unhealthy_cluster
        ]
    
    def plot_example_cluster(cluster, x_sample, is_factual):
    """
    This function 

    Input: a numpy array
    Output: a numpy array  
    """    
        
        center_color = "red"
        label="counterfactual"
        if is_factual:
            center_color = "green"
            label="factual"
            
        #plot the members of the cluster
        for i in range(len(cluster)):
            m=cluster[[c for c in cluster.columns 
                       if c.startswith("ts_")]].values[i:i+1][0]
            ax.plot(m, c="gray", alpha=0.25)

        # now plot the center of the cluster
        center=cluster[[c for c in cluster.columns 
                        if c.startswith("center_")]
                      ].values[i:i+1][0]
        ax.scatter(range(0, len(center)), center, c=center_color, label=label)
        ax.plot(center, "--", c=center_color)

        # now plot your selected sample
        ax.scatter(range(0, len(x_sample)), x_sample, s=30, c="orange",
                   label="predicted_sample")
        ax.plot(x_sample, "--", c="orange")

        ax.set_xticks(range(0,len(center)))
        ax.set_yticks(np.arange(0, 1, 0.1))
        ax.legend(loc="upper right")

        
    fig = plt.figure(figsize=(15,7))
    fig.suptitle("y_true={} | y_pred={} | confidence={}".format(x_actual_label,
                                                                x_prediction,
                                                                np.round(x_model_confidence,2)),
                 fontsize=20)


    ax = plt.subplot("121")
    ax.set_title("Factual_Cluster vs Sample",fontsize=16)
    plot_example_cluster(factual_cluster, x_sample, True)

    ax = plt.subplot("122")
    ax.set_title("Counterfactual_Cluster vs Sample", fontsize=16)
    plot_example_cluster(counterfactual_cluster, x_sample, False)

    plt.show()
    

def get_fft_values(y_values, T, N, f_s):
    """
    This function 

    Input: a numpy array
    Output: a numpy array  
    """    
    
    f_values = np.linspace(0.0, 1.0/(2.0*T), N) # N//2
    fft_values_ = fft(y_values)
    fft_values = 2.0/N * np.abs(fft_values_[0:N]) #N//2
    return f_values, fft_values

def extract_ffts(df):
    """
    This function 

    Input: a numpy array
    Output: a numpy array  
    """    
    
    
    ffts = list()
    frequencies= list()
    t_n=20
    T = t_n / df.shape[1]
    f_s = 1/T
    for x in df:
        f_values, fft_values =get_fft_values(x, T, df.shape[1], f_s)
        ffts.append(fft_values)
        frequencies.append(f_values)
    return np.array(ffts), np.array(frequencies)

def get_time_series_features(df, feature_name, w_size):
    """
    This function 

    Input: a numpy array
    Output: a numpy array  
    """    
    
    
    """Extracting time series standard deviation"""
    std_feature=df.copy()
    std_feature[feature_name]=df[feature_name].rolling(w_size
                                        ).std().fillna(
    df[feature_name][:w_size].std()).values
    
    std_x, std_y = cmapss.window_data(std_feature, w_size, w_size//2)
    
    
    """Extracting time series pitch"""
    max_feature=df.copy()
    max_feature[feature_name]=df[feature_name].rolling(w_size
                                        ).max().fillna(
    df[feature_name][:w_size].max()).values
    
    max_x, max_y = cmapss.window_data(max_feature, w_size, w_size//2)

    
    """Extracting time series min"""
    min_feature=df.copy()
    min_feature[feature_name]=df[feature_name].rolling(w_size
                                        ).min().fillna(
    df[feature_name][:w_size].min()).values
    
    min_x, min_y = cmapss.window_data(min_feature, w_size, w_size//2)
    
    
    """Extracting time series frequency"""
    X, y = cmapss.window_data(df, w_size, w_size//2)
    ffts, freq_bins = extract_ffts(X)
    
    
    """Extracting time series mean amplitute"""
    mean_amplitute_x = X
    
    
    """Concatenate time series features"""
    ts_features = np.concatenate((ffts, mean_amplitute_x, 
                                        std_x,max_x, min_x),
                                 axis=1)
    
    
    """Create a list of feature names"""
    feature_names = ["fft_{}".format(i) for i in range(0,w_size)]
    feature_names+=["mean_{}".format(i) for i in range(0,w_size)]
    feature_names+=["std_{}".format(i) for i in range(0,w_size)]
    feature_names+=["max_{}".format(i) for i in range(0,w_size)]
    feature_names+=["min_{}".format(i) for i in range(0,w_size)]
    
    return ts_features, feature_names

def extract_time_series_features(dev_data, feature_name, w_size):
    """
    This function 

    Input: a numpy array
    Output: a numpy array  
    """    
    
    
    dev_data = cmapss.get_univariate_cmapss(dev_data, feature_name)
    x_train, x_test = cmapss.train_test_split(dev_data)

    x_train = cmapss.minmax_scale(x_train)
    x_train = cmapss.denoise_sensors(x_train)

    x_test = cmapss.minmax_scale(x_test)
    x_test = cmapss.denoise_sensors(x_test)
    
    ts_feat_train, ts_feat_names = get_time_series_features(x_train, feature_name, w_size)
    ts_feat_test, ts_feat_names = get_time_series_features(x_test, feature_name, w_size)

    return ts_feat_train, ts_feat_test, ts_feat_names