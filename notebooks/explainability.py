import inspect_results as inspect
import prepare_data
from scipy.fft import fft, fftfreq
import numpy as np
import pandas as pd
from numpy import loadtxt

import tensorflow.keras as keras
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
np.random.seed(7)

cmapss = prepare_data.CMAPSS()
path = "../../Datasets/PHM08_Challenge_Data/train.txt"

def get_predictions_as_df(nd_array, y_true, y_pred):
    """
    This function creates a dataframe from the prediction results and the actual
    labels.

    Input: 3 numpy arrays: training data, actual labels and the predicted labels
    Output: the result dataframe  
    """    
    
    
    df = pd.DataFrame(data=nd_array)
    df.columns = ["ts_{}".format(i) for i in range(nd_array.shape[1])] 
    df["y_true"] = y_true
    df["y_pred"] = inspect.continues_to_binary(y_pred)
    df["confidence"] = y_pred
    
    return df
    

def plot_local_counter_and_factuals(train_cluster, train_cluster_df, selected_sample):
    """
    This function receives the train dataframe and a selected sample to 
    visualize the factual and counterfactual example. 
    The train_cluster is a 
    dictionary, which contains the trained clustering model and the 
    train_cluster_df is a dataframe, containing the cluster labels, actual 
    binary labels and the predicted labels with the prediction confidence.

    The function predicts the cluster of the given sample and identifies which
    factual cluster and which counterfactual cluster is the closest to this 
    sample. 

    Input: a dictionary, a dataframe, and one row of a dataframe
    Output: the visualization of the factual and counterfactual clusters 
    of the selected sample 
    """    
    
    # get the selected time series window (sequence) 
    # and exclude y_true, y_pred and model confidence
    x_sample=selected_sample[["ts_{}".format(i) for i in 
                              range(len(selected_sample)-4)]].values

    x_prediction = selected_sample["y_pred"]
    x_actual_label = selected_sample["y_true"]
    x_model_confidence = selected_sample["confidence"]

    # to which healthy cluster from the trainset is this sample the closest?
    closest_healthy_cluster = train_cluster["healthy_clusters"
                                           ]["kmeans"
                                            ].predict(x_sample[np.newaxis, :]
                                                      )[0]

    closest_unhealthy_cluster = train_cluster["unhealthy_clusters"
                                           ]["kmeans"
                                            ].predict(x_sample[np.newaxis, :]
                                                      )[0]
    
    closest_unhealthy_cluster += train_cluster["healthy_clusters"
                                              ]["labels"].max()+1
    
    center_cols = ["center_{}".format(i) for i in range(len(x_sample))]
    
    factual_cluster = None
    counterfactual_cluster = None

    factual_label = None
    counterfactual_label = None

    if int(x_prediction) == 1:
        # assuming your factual cluster is an unhealthy cluster
        factual_cluster = train_cluster_df[
            train_cluster_df.cluster==closest_unhealthy_cluster
        ]

        # and your counterfactual cluster is a healthy cluster
        counterfactual_cluster = train_cluster_df[
                train_cluster_df.cluster==closest_healthy_cluster
            ]
        
        factual_label = closest_unhealthy_cluster
        counterfactual_label = closest_healthy_cluster
    
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
        
        factual_label = closest_healthy_cluster
        counterfactual_label = closest_unhealthy_cluster
        
    cluster_labels = (factual_label, counterfactual_label)
    
    
    """ or training the local Tree """
    
    # prepare the data
    dev_mode = True 
    dev_data = cmapss.load_data(path)
    dev_data = cmapss.cluster_operational_settings(dev_data,
                                                   dev_mode)
    dev_data = cmapss.calculate_TTF(dev_data)
    dev_data = cmapss.calculate_continues_healthstate(dev_data)
    dev_data = cmapss.calculate_descrete_healthstate(dev_data)
    
    x_train = loadtxt('results/x_train.csv', delimiter=',')
    y_train = loadtxt('results/y_train.csv', delimiter=',')
    x_test = loadtxt('results/x_test.csv', delimiter=',')
    y_test = loadtxt('results/y_test.csv', delimiter=',')
    
    # extract train sets time series features
    dev_features = extract_time_series_features(dev_data,"s12", x_train.shape[1])
    
    ts_features_train, ts_features_test, feature_names = dev_features
    
    # now map the ts-features to their clusters
    ts_features_train_df = pd.DataFrame(data=ts_features_train, columns=feature_names)
    ts_features_train_df["y_pred"] = train_cluster_df.y_pred.values
    ts_features_train_df["y_true"] = train_cluster_df.y_true.values
    ts_features_train_df["confidence"] = train_cluster_df.confidence.values
    ts_features_train_df["cluster_conf"] = train_cluster_df.cluster_conf.values
    ts_features_train_df["cluster"] = train_cluster_df.cluster.values
    ts_features_train_df["unit"] = train_cluster_df.unit.values
   
    # now get the cluster of the ts-features data#
    factual_indices = ts_features_train_df[
        ts_features_train_df.cluster==factual_label].index

    counterfactual_indices = ts_features_train_df[
        ts_features_train_df.cluster==counterfactual_label].index
    
    # to create a surrogate model, load the nn model & get its predictions for the custer
    nn_model = keras.models.load_model("LSTM")
    lstm_pred_factuals = nn_model.predict(np.array(
        factual_cluster[[col for col in factual_cluster.columns 
                  if col.startswith("ts")]]
    )[:, :, np.newaxis])

    lstm_pred_counterfactuals = nn_model.predict(np.array(
        counterfactual_cluster[[col for col in counterfactual_cluster.columns 
                         if col.startswith("ts")]]
    )[:, :, np.newaxis])

    lstm_clusters_y_pred = np.concatenate((lstm_pred_factuals,
                                          lstm_pred_counterfactuals),
                                         axis=0)
    actual_clusters_y_true = np.concatenate((factual_cluster.y_true,
                                             counterfactual_cluster.y_true),
                                            axis=0)
    
    # now train your local DT
    local_dt = DecisionTreeClassifier(random_state=7, max_depth=5)
    local_dt.fit(ts_features_train[
        np.concatenate((factual_indices, counterfactual_indices), axis=0)],
                     inspect.continues_to_binary(lstm_clusters_y_pred))

    y_preds = local_dt.predict(ts_features_train[
        np.concatenate((factual_indices,
                        counterfactual_indices),
                       axis=0)])

    # calculate the feature importance for thess clusters
    importance = local_dt.feature_importances_
    f_imp = pd.DataFrame(importance, columns=["importance"])
    f_imp.index = feature_names
    f_imp["importance"] = f_imp.importance.values * 100
    f_imp=f_imp.sort_values(by=['importance'], ascending=False)
    display(f_imp.head(5))
    
    # look at the decision cuts for thess clusters
    display(tree_to_code(local_dt, feature_names))
    
    
    
    def plot_example_cluster(cluster, x_sample, is_unhealthy, is_counterfactual):
        """
        This function receives a cluster, the predicted sample, and a boolean 
        value to assign a color to the cluster subplot (a factual cluster centroid 
        is plotted with green and the centroid of the counterfactual cluster is 
        plotted with red).

        Input: a dataframe, a numpy array, and a boolean value
        Output: subplot of the cluster
        """    
        
        center_color = "green"
        label = "factual"
        if is_unhealthy and is_counterfactual:
            center_color = "green" 
            label="counterfactual"

        elif not is_unhealthy and is_counterfactual:
            center_color = "red"

        elif is_unhealthy and not is_counterfactual:
            label="factual"
            center_color = "red" 
            
            
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
        ax.scatter(range(0, len(x_sample)), x_sample, s=30, c="black",
                   label="predicted_sample")
        ax.plot(x_sample, "--", c="black")

        ax.set_xticks(range(0,len(center)))
        ax.set_yticks(np.arange(0, 1, 0.1))
        ax.legend(loc="lower right")

        
    fig = plt.figure(figsize=(15,5))
    fig.suptitle(
        "y_true={} | y_pred={} | Sigmoid-Value={}".format(x_actual_label,
                                                       x_prediction,
                                                       np.round(
                                                           x_model_confidence,2
                                                           )),
                 fontsize=20)


    ax = plt.subplot("121")
    ax.set_title("Factual_Cluster vs Sample",fontsize=16)
    plot_example_cluster(factual_cluster, x_sample, x_prediction, False)

    ax = plt.subplot("122")
    ax.set_title("Counterfactual_Cluster vs Sample", fontsize=16)
    plot_example_cluster(counterfactual_cluster, x_sample, x_prediction, True)

    plt.show()
    
    
def get_counter_and_factuals(train_cluster, train_cluster_df, selected_sample):
    # get the selected time series window (sequence) 
    # and exclude y_true, y_pred and model confidence
    x_sample=selected_sample[["ts_{}".format(i) for i in 
                              range(len(selected_sample)-4)]].values

    x_prediction = selected_sample["y_pred"]
    x_actual_label = selected_sample["y_true"]
    x_model_confidence = selected_sample["confidence"]

    # to which healthy cluster from the trainset is this sample the closest?
    closest_healthy_cluster = train_cluster["healthy_clusters"
                                           ]["kmeans"
                                            ].predict(x_sample[np.newaxis, :]
                                                      )[0]

    closest_unhealthy_cluster = train_cluster["unhealthy_clusters"
                                           ]["kmeans"
                                            ].predict(x_sample[np.newaxis, :]
                                                      )[0]

    # now I have to change the unhealthy cluster label 
    # this is because I did the same when creating the cluster DF .csv file
    # and wanted to avoid duplicate cluster labels for healthy and unhealthy
    closest_unhealthy_cluster += train_cluster["healthy_clusters"
                                              ]["labels"].max()+1
    
    center_cols = ["center_{}".format(i) for i in range(len(x_sample))]
    
    factual_cluster = None
    counterfactual_cluster = None
    
    factual_label = None
    counterfactual_label = None

    if int(x_prediction) == 1:
        # assuming your factual cluster is an unhealthy cluster
        factual_cluster = train_cluster_df[
            train_cluster_df.cluster==closest_unhealthy_cluster
        ]

        # and your counterfactual cluster is a healthy cluster
        counterfactual_cluster = train_cluster_df[
                train_cluster_df.cluster==closest_healthy_cluster
            ]
        
        factual_label = closest_unhealthy_cluster
        counterfactual_label = closest_healthy_cluster
    
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
        
        factual_label = closest_healthy_cluster
        counterfactual_label = closest_unhealthy_cluster
        
    cluster_labels = (factual_label, counterfactual_label)
    return factual_cluster, counterfactual_cluster, cluster_labels, int(x_prediction)


def plot_example_cluster(ax, cluster, c_label, x_sample, is_unhealthy, is_counterfactual, x_lim):    
        
    center_color = "green"
    sample_color = "green"
    if is_unhealthy and is_counterfactual:
        center_color = "green" 
        sample_color = "red"
        
    elif not is_unhealthy and is_counterfactual:
        center_color = "red"
    
    elif is_unhealthy and not is_counterfactual:
        sample_color = "red"  
        center_color = "red" 
                 
    for row in range(len(cluster)):
        m=cluster[[col for col in cluster.columns 
                   if col.startswith("ts_")]].values[row:row+1][0]
        
        ax.plot(x_lim, m, c="gray", alpha=0.25)  
    
    # now plot the center of the cluster
    center=cluster[[col for col in cluster.columns 
                    if col.startswith("center_")]
                  ].values[0]

    ax.scatter(x_lim, center, c=center_color, label="")
    ax.plot(x_lim, center, "--", c=center_color)
    ax.text(x_lim[len(x_lim)//2 - 3], center[len(center)//2]+0.1,
             "cluster {}".format(c_label), fontsize = 16, c=center_color)

    ax.plot(x_lim, x_sample, c="black")
    
    

def explain_engine(engine_parameters, engine_num):
    
    train_cluster = engine_parameters["train_cluster"]
    train_cluster_df = engine_parameters["train_cluster_df"]
    x_test = engine_parameters["x_test"]
    y_test = engine_parameters["y_test"]
    lstm_pred_test = engine_parameters["lstm_pred_test"]
    
    x_engine=[]
    conf_engine=[]
    y_engine = []
    y_hat_engine = []
    
    test_df = pd.read_csv("results/test_df.csv", sep=',')
    test_engines = test_df.unit.values
    test_df = get_predictions_as_df(x_test, y_test, lstm_pred_test)
    test_df["unit"] = test_engines

    #-1 = unit #-2 = conf #-3 = pred # -4 = true
    for i in range(len(test_df[test_df.unit==engine_num])):
        
        if i%2==1: continue
        
        engine_data = test_df[test_df.unit==engine_num].reset_index(drop=True)
        x_engine += list(engine_data.loc[i][:-4])

        conf_engine += [engine_data.loc[i][-2] for _ in range(20)]
        y_hat_engine += [engine_data.loc[i][-3] for _ in range(20)]
        y_engine += [engine_data.loc[i][-4] for _ in range(20)]
    
    plt.figure(figsize=(20,5))
    plt.title("Engine {} Actual Labels".format(engine_num), 
             fontsize=20)
    plt.plot(x_engine, c="black", label="Engine {}".format(engine_num))
    plt.plot(conf_engine, c="gray", label="Sigmoid Output".format(engine_num))
    
    for i in range(len(y_engine)):
        if y_engine[i]==0:
            plt.axvspan(xmin=i, xmax=i+1, ymax=1,
                        facecolor='green', alpha=0.2)
        else:
            plt.axvspan(xmin=i, xmax=i+1, ymax=1,
                        facecolor='red', alpha=0.2)    
    plt.ylim(0,1.1,0.2)
    plt.legend(fontsize=14)        
    plt.savefig("./results/e{}_y_true.png".format(engine_num))
    
    """#################### now the explanation plot ####################"""
    fig, axs = plt.subplots(2, sharex=True, figsize=(20,10))
    fig.suptitle("Engine {} Predicted Labels".format(engine_num), 
             fontsize=20)
    
    is_first_round = True
    for i in range(len(engine_data)):
        
        if i%2==1: continue
        
        print("=", end="")
        selected_sample = engine_data.loc[i]

        x_sample = list(engine_data.loc[i][:-4])
        factuals, counterfactuals, labels, x_pred = get_counter_and_factuals(train_cluster,
                                                                    train_cluster_df,
                                                                    selected_sample)

        factual_clabel = labels[0]
        counterfactual_clabel = labels[1]
        
        if is_first_round:
            x_axis_loc = 0
            vertical_lim = [vl for vl in range(20)] # as long as the sequence length
        else:
            x_axis_loc +=20
            vertical_lim = [x_axis_loc+vl for vl in range(20)]  

        plot_example_cluster(axs[0], factuals,factual_clabel, 
                             x_sample, x_pred, False, vertical_lim)
        plot_example_cluster(axs[1], counterfactuals, counterfactual_clabel,
                             x_sample, x_pred, True, vertical_lim)

        is_first_round=False

    # give background color for the predicted labels
    for i in range(len(y_hat_engine)):
        
        if y_hat_engine[i]==0:
            axs[0].axvspan(xmin=i, xmax=i+1, ymax=1,
                        facecolor='green', alpha=0.2)
            axs[1].axvspan(xmin=i, xmax=i+1, ymax=1,
                        facecolor='green', alpha=0.2)
        else:
            axs[0].axvspan(xmin=i, xmax=i+1, ymax=1,
                        facecolor='red', alpha=0.2) 
            axs[1].axvspan(xmin=i, xmax=i+1, ymax=1,
                        facecolor='red', alpha=0.2) 
        
    axs[0].set_title("Factual Examples", fontsize=20)
    axs[0].set_xticks(np.arange(0, max(vertical_lim), 20))
    axs[0].set_yticks(np.arange(0, 1.1, 0.1))

    axs[1].set_title("CounterFactual Examples", fontsize=20)
    axs[1].set_xticks(np.arange(0, max(vertical_lim), 20))
    axs[1].set_yticks(np.arange(0.3, 1.1, 0.1))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    plt.show()
    plt.savefig("./results/e{}_explanation.png".format(engine_num))


def get_fft_values(y_values, T, N, f_s):
    """
    This function extracts the double-sided FFT values of the input time series,
    y_values based on the sampling period (T), time series length (N), and 
    the sampling frequency (f_s).
    The reason behind calculating the double-sided FFT values is to keep preserve
    the input shape.


    Input: a numpy array, a float number, an integer, and another float number
    Output: 2 arrays of extracted frequency bins and FFT values
    """    
    
    f_values = np.linspace(0.0, 1.0/(2.0*T), N) # N//2
    fft_values_ = fft(y_values)
    fft_values = 2.0/N * np.abs(fft_values_[0:N]) #N//2
    return f_values, fft_values

def extract_ffts(nd_array):
    """
    This function receives a 2d array of the full time series and using 
    the get_fft_values function, extracts the FFT values.

    Input: a numpy array
    Output: two numpy arrays
    """    
    
    
    ffts = list()
    frequencies= list()
    t_n=20
    T = t_n / nd_array.shape[1]
    f_s = 1/T
    for x in nd_array:
        f_values, fft_values =get_fft_values(x, T, nd_array.shape[1], f_s)
        ffts.append(fft_values)
        frequencies.append(f_values)
    return np.array(ffts), np.array(frequencies)

def get_time_series_features(df, feature_name, w_size):
    """
    This function extracts the time domain and the frequency domain features of
    each w_size time window from the given feature name (a column existing in 
    the dataframe). 
    

    Input: a dataframe, a string, and an integer
    Output: a n dimentional array and a list of strings  
    """    
    
    
    """Extracting time series standard deviation"""
    std_feature=df.copy()
    std_feature[feature_name]=df[feature_name].rolling(w_size
                                        ).std().fillna(
    df[feature_name][:w_size].std()).values
    
    std_x, std_y, engines = cmapss.window_data(std_feature, w_size, w_size//2)
    
    
    """Extracting time series pitch"""
    max_feature=df.copy()
    max_feature[feature_name]=df[feature_name].rolling(w_size
                                        ).max().fillna(
    df[feature_name][:w_size].max()).values
    
    max_x, max_y, engines = cmapss.window_data(max_feature, w_size, w_size//2)

    
    """Extracting time series min"""
    min_feature=df.copy()
    min_feature[feature_name]=df[feature_name].rolling(w_size
                                        ).min().fillna(
    df[feature_name][:w_size].min()).values
    
    min_x, min_y, engines = cmapss.window_data(min_feature, w_size, w_size//2)
    
    
    """Extracting time series frequency"""
    X, y, engines = cmapss.window_data(df, w_size, w_size//2)
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
    This function uses the get_time_series_feature function and extracts the
    time series features from the given train and test data. And returns the
    feature sets with their corresponding list of feature names. 

    Input: a dataframe
    Output: 2 numpy arrays and a list of strings
    """    
    
    
    dev_data = cmapss.get_univariate_cmapss(dev_data, feature_name)
    x_train, x_test = cmapss.train_test_split(dev_data)

    x_train = cmapss.minmax_scale(x_train)
    x_train = cmapss.denoise_sensors(x_train)

    x_test = cmapss.minmax_scale(x_test)
    x_test = cmapss.denoise_sensors(x_test)
    
    ts_feat_train, ts_feat_names = get_time_series_features(x_train,
                                                            feature_name,
                                                            w_size)
    ts_feat_test, ts_feat_names = get_time_series_features(x_test,
                                                           feature_name,
                                                           w_size)

    return ts_feat_train, ts_feat_test, ts_feat_names


def tree_to_code(tree, feature_names):
    
    from sklearn.tree import _tree
    
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    #print("def tree({}):".format(", ".join(feature_names)))

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print("{}if {} <= {}:".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            print("{}else:  # if {} > {}".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
        else:
            print("{}return {}".format(indent, np.argmax(tree_.value[node][0])))

    recurse(0, 1)