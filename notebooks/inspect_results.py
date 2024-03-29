from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier


from sklearn.cluster import KMeans # only for spatial dimension
from tslearn.clustering import TimeSeriesKMeans
from sklearn.metrics import silhouette_score
from scipy.spatial import distance

import numpy as np
import pandas as pd
np.random.seed(7)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

def plot_classification_report(y_true, y_pred):
    """
    This function receives two arrays, y_pred and y_treu, and prints out the 
    classification report, It also calculates the confusion matrix.
    
    Input: two NumPy arrays, y_true, and y_pred
    Output: the classification report and the plot of the confusion matrix 
    """
    
    
    print(classification_report(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred,
                          labels=[0, 1])
    plt_show = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=[0, 1])  
    plt_show.plot()
    plt.show()
    
def continues_to_binary(y_pred):
    """
    This function transforms a list of continuous values into binary values. 
    Used for the predictions of my deep learning models.

    Input: a numpy array
    Output: a numpy array  
    """
    return [1 if i>0.5 else 0 for i in y_pred]

def DTW(a, b):   
    """
    This function calculates the distance with Dynamic Time Warping
    
    Input: two numpy arraywith similar dimensions
    Output: a matrix of warping distance
    """
    
    
    an = a.size
    bn = b.size
    pointwise_distance = distance.cdist(a.reshape(-1,1),b.reshape(-1,1))
    cumdist = np.matrix(np.ones((an+1,bn+1)) * np.inf)
    cumdist[0,0] = 0

    for ai in range(an):
        for bi in range(bn):
            minimum_cost = np.min([cumdist[ai, bi+1],
                                   cumdist[ai+1, bi],
                                   cumdist[ai, bi]])
            cumdist[ai+1, bi+1] = pointwise_distance[ai,bi] + minimum_cost

    return cumdist[an, bn]

def KNN_DTW(X, y):
    """
    This function classifies the X based on the given y.

    Input: X input array
    Output: a knn model trained on X w.r.t. y
    """
    
    # number of neighbours was only tuned on CMAPSS dataset 
    # with the current parameters
    knn = KNeighborsClassifier(metric=DTW, n_neighbors=4).fit(X, y)
    return knn
    
    
def cluster_data(X, n_clusters):
    """
    This function receives an array and an integer to cluster the array into 
    n_clusters.

    Input: an array and an integer
    Output: a dictionary of the cluster model, list of the labels and their 
    corresponding centroids
    """    
    # time and spatial dimensions are taken into acount
    kmeans = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw",
                              max_iter=10, random_state=7)
    
    # only spatial dimension is taken into acount: not recommended
    #kmeans = KMeans(n_clusters = n_clusters, random_state = 7)
    
    labels = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_
    return {"kmeans":kmeans,
            "labels":labels, 
            "centroids":centroids}
    
def find_the_best_n_cluster(X):
    """
    This function applies the cluster_data multiple times to return the 
    best number of clusters for the given data. I started first with smaller 
    numbers for the clusters. But as I continued with the experiment, I found 
    the need to have more clusters. This is because I want to create smaller 
    neighborhoods, and it is only possible when the number of clusters is 
    larger. However, comparing the silhouette_scores, increasing 
    the n_clusters does not decrease the accuracy of the clustering model.

    Input: a numpy array
    Output: an integer
    """        
    

    # Find the appropriate n_clusters for each class  
    scores=[]
    
    #range_n_clusters = [3,5,7,10,15,20,25,30]
    
    # I avoided the smaller n_clusters to have smaller neighbourhoods
    range_n_clusters = [15,20,25,30]
    for n_clusters in range_n_clusters:
        cluster_dict = cluster_data(X, n_clusters)
        silhouette_avg = silhouette_score(X, cluster_dict["labels"])
        scores.append(silhouette_avg)
    
    selected_n_cluster=range_n_clusters[scores.index(np.max(scores))]
    return selected_n_cluster

def visualize_clusters(X, cluster, title):
    """
    This function visualized the input clusters of the given data X. Since we 
    have high dimensional data (each time point of a sequence is considered a 
    feature), I only choose the first and last steps to visualize the clusters 
    in a two-dimensional plot.

    Input: a numpy array, a dictionary of cluster model, labels and the 
    centroids, and a string for the title of the plot
    
    Output: plot of the clustered X
    """    
    
    
    f1 = 0 # visulizing timestep f1
    f2 = 19 # over the timestep f2
    u_labels = np.unique(cluster["labels"])

    for l in u_labels:
        plt.scatter(X[cluster["labels"] == l , f1],
                    X[cluster["labels"]== l , f2],
                    label = l, alpha=0.05)
    plt.scatter(cluster["centroids"][:,f1],
                cluster["centroids"][:,f2],
                color = 'k')

    plt.title(title, fontsize=16)
    plt.ylim(0,1,0.1);plt.xlim(0,1,0.1)
    plt.ylabel("timestep {}".format(f1), fontsize=12)
    plt.xlabel("timestep {}".format(f2), fontsize=12)
    plt.show()

def get_clustered_data(nd_array, y, is_y_pred):
    """
    This function uses the cluster_data function to cluster the given array based 
    on the given labels. 
    The labels could be either the actual labels or the predicted labels within 
    each of its classes. 
    
    We cluster the data to see the existing patterns or sub-classes of a class. 
    We believe this information can help us understand the small changes 
    that could lead to model misclassification.

    Input: a numpy array, labels, and a boolean variable indicating whether the 
    label is a prediction or true. 
    Output: a dictionary of clustered data for each class and their cluster
    information. 
    """    
    
    
    label="y_true"
    if is_y_pred:
        label="y_pred"
    
    df = pd.DataFrame(data=nd_array)
    df.columns = ["ts_{}".format(i) for i in range(nd_array.shape[1])] 
    df[label] = y

    x_0 = df.loc[df[label] == 0, df.columns != label].values
    x_1 = df.loc[df[label] == 1, df.columns != label].values    

    # Find the best number for clusters and cluster the data
    cluster_0 = cluster_data(x_0, find_the_best_n_cluster(x_0))
    cluster_1 = cluster_data(x_1, find_the_best_n_cluster(x_1))
    
    return {"healthy_data":x_0, 
            "healthy_clusters":cluster_0,
            "unhealthy_data":x_1,
            "unhealthy_clusters":cluster_1}


def get_clustered_df(nd_array, y_true, y_pred):
    """
    This function uses clustered_data and clusters the classes of the input 
    array. The difference between this function and the previous one is that
    this function aims to return a dataframe with both y_pred and y_true and 
    the model confidence of the prediction.

    Input: 3 numpy arrays= x, true labels, and the predicted labels
    Output: the prediction result dataframe 
    """    
    
    
    df = pd.DataFrame(data=nd_array)
    df.columns = ["ts_{}".format(i) for i in range(nd_array.shape[1])] 
    
    # add the prediction results
    df["y_pred"] = [1 if i>0.5 else 0 for i in y_pred]

    x_0 = df.loc[df["y_pred"] == 0, df.columns != "y_pred"].values
    x_1 = df.loc[df["y_pred"] == 1, df.columns != "y_pred"].values    

    # Find the best number for clusters and cluster the data
    cluster_0 = cluster_data(x_0, find_the_best_n_cluster(x_0))
    cluster_1 = cluster_data(x_1, find_the_best_n_cluster(x_1))

    df["y_true"] = y_true 
    
    #add the confidence
    df["confidence"] = y_pred


    # add the cluster labels
    df.loc[df[df.y_pred==0].index, "cluster"] = cluster_0["labels"]
    df.loc[df[df.y_pred==1].index, "cluster"] = (cluster_0["labels"].max()+1
                                                ) + cluster_1["labels"]
    df.cluster = df.cluster.astype(int)


    # add cluster centroids
    feature_length = nd_array.shape[1]

    for i in range(feature_length):
        df["center_{}".format(i)] = np.nan

        for cluster in np.unique(df.cluster):

            for j in range(len(cluster_0["centroids"])):
                if cluster == j: 
                    df.loc[df[df.cluster==cluster].index,
                    "center_{}".format(i)] = cluster_0["centroids"][j][i][0] 
            for j in range(len(cluster_1["centroids"])):
                if cluster == cluster_0["labels"].max()+1+j: 
                    df.loc[df[df.cluster==cluster].index,
                    "center_{}".format(i)] = cluster_1["centroids"][j][i][0] 


    # add cluster confidence
    df['cluster_conf'] = df.groupby('cluster')['confidence'].transform('mean')

    return df


def label_point(x, y, val, ax):
    """
    This function includes the cluster labels to the cluster plot 
    """    
    
    
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x']+.015, point['y']+.015,
                str(point['val'].astype("int")),
                size=16, color="black")
        
        
def visualize_predictions(plot_df, model_name):
    """
    This function visualizes the clustered prediction result. 

    Input: a dataframe and the model_name for the plot title
    Output: the visualization of the input dataframe 
    """    
    
    
    # visulalize the first timesteps over the last timesteps
    f1 = "ts_0"
    f2 = "ts_19"

    fc1 = f1.replace("ts", "center")
    fc2 = f2.replace("ts", "center")
    
    # add confidence to the color = lower confidence, lighter the color
    color_0 = np.array([(18/201, 27/201, 1, alpha) 
    for alpha in plot_df[plot_df.y_true == 0].cluster_conf.values]) 
    color_1 = [(1, 0.4, 0, alpha) 
    for alpha in plot_df[plot_df.y_true == 1].cluster_conf.values]

    plt.figure(figsize=(10,7))
    plt.title("y_true VS y_pred ({})".format(model_name),fontsize=20)

    plt.ylabel(f1, fontsize=16)
    plt.xlabel(f2, fontsize=16)

    plt.ylim(0,1,0.1);plt.xlim(0,1,0.1)

    # add the cluster labels to the plot
    label_point(plot_df[fc1], plot_df[fc2],
                plot_df["cluster"], plt.gca())

    sns.scatterplot(data=plot_df,
                    x=f1, y=f2,
                    hue="y_pred", style="y_true",
                    s=50, alpha=0.025)
    sns.scatterplot(data=plot_df, 
                    x=fc1, y=fc2,
                    hue="y_pred", style="y_true",
                    s= 100, sizes="cluster_conf")
    
    plt.savefig("./results/decision_boundary.png")