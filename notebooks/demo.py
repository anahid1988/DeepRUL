import streamlit as st
import numpy as np
import pandas as pd

#ML
import tensorflow.keras as keras
import pickle
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier

# my own libraries
import sys
sys.path.insert(0, './')
import prepare_data
import inspect_results as inspect
import explainability as explainer

# for visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly_express as px
from PIL import Image

import warnings
warnings.filterwarnings("ignore")
np.random.seed(7)

################################################################################
# functions
################################################################################
def add_logo(logo_path, width, height):
    """Read and return a resized logo"""
    logo = Image.open(logo_path)
    modified_logo = logo.resize((width, height))
    return modified_logo


def load_data(p, mode):
    data = cmapss.load_data(path)
    data = cmapss.cluster_operational_settings(data, mode)
    data = cmapss.calculate_TTF(data)
    data = cmapss.calculate_continues_healthstate(data)
    data = cmapss.calculate_descrete_healthstate(data)

    return data


def interactive_plot(df):
    col1, col2 = st.columns(2)
    
    engine_id = col1.selectbox('Select the engine ID', options=df.unit.unique())
    x_axis_val = "cycle"
    y_axis_val = col2.selectbox('Select the Y-axis', options=df.columns[2:])

    title="Interactive plot"
    names = {
        "s1":"Total Tempreture at fan inlet R",
        "s2":"Total Tempreture at LPC inlet R (LPC= Low-pressure Compressor)",
        "s3":"Total Tempreture at HPC inlet R (LPC= High-pressure Compressor)",
        "s4":"Total Tempreture at LPT inlet R (LPC= High-pressure Turbin)",
        "s5":"Total Pressure at fan inlet psia",
        "s6":"Total Pressure in bypass-duct psia",
        "s7":"Total Pressure at the HPC outlet psia",
        "s8":"Physical fan speed rpm",
        "s9":"Physical core speed rpm",
        "s10":"Engine pressure ratio (P50/P2) (unitless)",
        "s11":"Static pressure at HPC",
        "s12":"Ratio of fuel flow to Ps30 pps/psi",
        "s13":"Corrected fan speed rpm",
        "s14":"Corrected core speed rpm",
        "s15":"Bypass Ratio (unitless)",
        "s16":"Burner fuel-air ratio (unitless)",
        "S17":"Bleed Enthalpy (unitless)",
        "s18":"Demanded fan speed rpm",
        "s19":"Demanded corrected fan speed rpm",
        "s20":"HPT coolant bleed lbm/s (HPT = high-pressure turbin)",
        "s21":"LPT coolant bleed lbm/s (LPT = low-pressure turbin)"
        }

    if y_axis_val in names.keys():
        title=names.get(y_axis_val)

    plot = px.line(df[df.unit==engine_id], x=x_axis_val, y=y_axis_val,
                   title=title)
    plot.update_traces(mode='markers+lines')
    st.plotly_chart(plot, use_container_width=True)


def report_baseline(model_name, x, y):
    with open('../results/{}.pkl'.format(model_name), 'rb') as f:
        sk_model = pickle.load(f)
    f.close()

    y_hat = sk_model.predict(x)
    cm = confusion_matrix(y, y_hat,
                            labels=[0, 1])
    return y_hat, cm


def load_lstm(name):
    model = keras.models.load_model("../results/{}.hdf5".format(name))
    return model


def load_cluster_models(file_name):
    cluster_df = pd.read_csv("../results/{}_df.csv".format(file_name), sep=',')

    with open('../results/{}.pkl'.format(file_name), 'rb') as f:
        cluster = pickle.load(f)
    f.close()

    return cluster_df, cluster


################################################################################
# end functions
# start the demo
################################################################################

# Title of the Demo
st.title("""eXplainable Deep Neural Networks for Machine Health Prognosis
applications""")

my_logo = add_logo(logo_path="../figures/XPDM-LOGO.png",width=150,height=150)
st.sidebar.image(my_logo)
st.sidebar.title("eXplainable PdM")
st.sidebar.markdown("Is the turbofan engine healthy or unhealthy?")

options = st.sidebar.radio('## Display:', ['Home',
                                           'Display dataframe',
                                           'Plot data',
                                           'Plot health labels',
                                           'Plot processed data',
                                           'Baseline-models report',
                                           'Deep Model (LSTM) report',
                                           'Explain the classifier (Global)',
                                           'Explain the classifier (Cohort)'])

path = "../data/train.txt"
dev_mode = True # not using the CMAPSS test set
nn_model = None

cmapss = prepare_data.CMAPSS()
dev_data = load_data(path, dev_mode)

x_train = np.loadtxt('../results/x_train.csv', delimiter=',')
y_train = np.loadtxt('../results/y_train.csv', delimiter=',')
x_test = np.loadtxt('../results/x_test.csv', delimiter=',')
y_test = np.loadtxt('../results/y_test.csv', delimiter=',')

train_cluster_df, train_cluster = load_cluster_models("train_cluster")



if options == 'Home':  
    st.subheader('Motivation and Problem Statement.')
    st.write("""Interpretable machine learning has recently attracted a lot of
    interest in the community. The current explainability approaches mainly
    focus on models trained on non-time series data.""")
    st.write("""LIME and SHAP are well-known post-hoc examples that provide
    visual explanations of feature contributions to model decisions on an
    instance basis.""")
    st.write("""Other approaches, such as attribute-wise interpretations,
    only focus on tabular data. Little research has been done so far on the
    interpretability of predictive models trained on time series data.""")

    image_file = add_logo(logo_path="../figures/CMAPSS_description.png",width=750,height=500)
    st.image(image_file)


if options == 'Display dataframe': 


    st.subheader('CMAPSS Data')
    st.write('Data Source: https://data.nasa.gov/widgets/xaut-bemq')
    st.write("""Commercial Modular Aero-Propulsion System Simulation is intended
     for Prognosis Health Management tasks such as Time-To-Failure (TTF) and
     Remaining-Useful-Life (RUL) estimation.""")

    st.write(dev_data)

if options == 'Plot data':
    st.write("Please choose an engine and the parameter to visualize sensor data over the number of flights.")
    interactive_plot(dev_data)

if options=='Plot health labels':
    
    st.write('Calculated Engine Health Status:')
    image_file = "../results/figures/true_healthstatus.png"
    st.image(image_file)

if options=='Plot processed data':
    dev_data = cmapss.get_univariate_cmapss(dev_data, "s12")
    x_train, x_test = cmapss.train_test_split(dev_data)
    x_train = cmapss.minmax_scale(x_train)
    x_train = cmapss.denoise_sensors(x_train)

    st.write("Please choose an engine and the parameter to visualize dsensor data over the number of flights.")
    interactive_plot(x_train)

if options =='Baseline-models report':
    choose_model = st.sidebar.selectbox("Choose the Baseline Model",
    	["NONE",
        "LogisticRegressionCV",
        "RidgeClassifierCV",
        "KNeighborsClassifier",
        "DecisionTreeClassifier"])

    if(choose_model != "NONE"):
        y_pred, cf_matrix = report_baseline(choose_model, x_test, y_test)

        st.write('\n{} Classification Report:\n'.format(choose_model))
        class_names=["healthy", "unhealthy"]

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.write("Accuracy: {}%".format(
                accuracy_score(y_test, y_pred
                               ).round(4)*100))

        with col2:
            st.write("Precision: {}%".format(
                precision_score(y_test, y_pred,labels=class_names
                                ).round(4)*100))

        with col3:
            st.write("Recall: {}%".format((
                recall_score(y_test, y_pred, labels=class_names
                             ).round(4)*100).round(2)))

        with col4:
            st.write("F1 Score: {}%".format(
                f1_score(y_test, y_pred, labels=class_names
                         ).round(4)*100))


        st.subheader('Confusion Matrix:')
        st.write(cf_matrix)
        fig, ax = plt.subplots(figsize=(3,2))
        sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True,
                fmt='.2%', cmap='Blues')
        st.pyplot(fig)

if options =='Deep Model (LSTM) report':
    nn_model = load_lstm("lstm")

    lstm_pred_test = nn_model.predict(x_test[:, :, np.newaxis])
    cm = confusion_matrix(y_test,
                          inspect.continues_to_binary(lstm_pred_test),
                          labels=[0, 1])

    st.write('Model Summary:')
    image_file = "../results/figures/LSTM_Summary.png"
    st.image(image_file)

    st.write('LSTM Classification Report:\n')

    class_names=["healthy", "unhealthy"]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.write("Accuracy: {}%".format(accuracy_score(y_test,
    inspect.continues_to_binary(lstm_pred_test)
    ).round(4)*100))

    with col2:
        st.write("Precision: {}%".format(precision_score(y_test,
    inspect.continues_to_binary(lstm_pred_test),
    labels=class_names).round(4)*100))

    with col3:
        st.write("Recall: {}%".format(
        (recall_score(y_test,
        inspect.continues_to_binary(lstm_pred_test),
         labels=class_names).round(4)*100
        ).round(2)))

    with col4:
        st.write("F1 Score: {}%".format((f1_score(
                y_test,
                inspect.continues_to_binary(lstm_pred_test),
                labels=class_names).round(4)*100).round(2)))

    st.subheader('Confusion Matrix:')
    st.write(cm)
    fig, ax = plt.subplots(figsize=(3,2))
    sns.heatmap(cm/np.sum(cm), annot=True,
            fmt='.2%', cmap='Blues')
    st.pyplot(fig)

if options=="Explain the classifier (Global)":
    if not nn_model:
        nn_model = load_lstm("lstm")

        lstm_pred_test = nn_model.predict(x_test[:, :, np.newaxis])

    sub_options = st.sidebar.radio('## Display:', ['Surrogate Tree',
                                                   'Model Decision Boundary'])

    if sub_options=='Surrogate Tree':
        dev_features = explainer.extract_time_series_features(dev_data,
                                                              "s12",
                                                              x_train.shape[1])

        ts_features_train, ts_features_test, feature_names = dev_features


        lstm_pred_train = nn_model.predict(x_train[:, :, np.newaxis])

        surrogate_dt = DecisionTreeClassifier(random_state=7, max_depth=10)
        surrogate_dt.fit(ts_features_train,
                         inspect.continues_to_binary(lstm_pred_train))

        surrogate_dt_train_preds = surrogate_dt.predict(ts_features_train)
        surrogate_dt_test_preds = surrogate_dt.predict(ts_features_test)

        st.write('Surrogate Tree Classification Report:\n')

        class_names=["healthy", "unhealthy"]

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.write("Accuracy: {}%".format(accuracy_score(y_test,
        inspect.continues_to_binary(lstm_pred_test)
        ).round(4)*100))

        with col2:
            st.write("Precision: {}%".format(precision_score(y_test,
        inspect.continues_to_binary(lstm_pred_test),
        labels=class_names).round(4)*100))

        with col3:
            st.write("Recall: {}%".format(
            (recall_score(y_test,
            inspect.continues_to_binary(lstm_pred_test),
             labels=class_names).round(4)*100
            ).round(2)))

        with col4:
            st.write("F1 Score: {}%".format((f1_score(
                    y_test,
                    inspect.continues_to_binary(lstm_pred_test),
                    labels=class_names).round(4)*100).round(2)))

        st.subheader('Confusion Matrix:')
        sdt_cm = confusion_matrix(inspect.continues_to_binary(lstm_pred_test),
                              surrogate_dt_test_preds,
                              labels=[0, 1])

        st.write(sdt_cm)
        fig, ax = plt.subplots(figsize=(3,2))
        sns.heatmap(sdt_cm/np.sum(sdt_cm), annot=True,
                fmt='.2%', cmap='Blues')
        st.pyplot(fig)

        st.write("## Most Influential Features:")
        importance = surrogate_dt.feature_importances_
        # summarize feature importance
        f_imp = pd.DataFrame(importance, columns=["importance"])
        f_imp.index = feature_names
        f_imp["importance"] = f_imp.importance.values * 100

        f_imp=f_imp.sort_values(by=['importance'], ascending=False)
        st.write(f_imp.head(10))

        # plot feature importance
        st.bar_chart(f_imp[:10])

        st.write('### Surrogate Tree Decision Rules:')
        image_file = "../results/figures/surrogate_tree.png"
        st.image(image_file)

    if sub_options=="Model Decision Boundary":
        col1, col2 = st.columns(2)

        with col1:
            st.write("""All the predictions:""")
            inspect.visualize_predictions(train_cluster_df, "LSTM")
            image_file = "../results/decision_boundary.png"
            st.image(image_file)

        with col2:
            st.write("""Failed Predictions:""")
            inspect.visualize_predictions(train_cluster_df[
            train_cluster_df.y_true != train_cluster_df.y_pred],
                          "LSTM")
            image_file = "../results/decision_boundary.png"
            st.image(image_file)

        col3, col4 = st.columns(2)
        with col3:
            st.write("""False Positives:""")
            inspect.visualize_predictions(train_cluster_df[
            (train_cluster_df.y_true==0) & (train_cluster_df.y_pred==1)],
                          "LSTM")
            image_file = "../results/decision_boundary.png"
            st.image(image_file)

        with col4:
            st.write("""False Negatives:""")
            inspect.visualize_predictions(train_cluster_df[
            (train_cluster_df.y_true==1) & (train_cluster_df.y_pred==0)],
                          "LSTM")
            image_file = "../results/decision_boundary.png"
            st.image(image_file)


if options=="Explain the classifier (Cohort)":
    if not nn_model:
        nn_model = load_lstm("lstm")

    lstm_pred_test = nn_model.predict(x_test[:, :, np.newaxis])

    test_df = pd.read_csv("../results/test_df.csv", sep=',')
    test_engines = test_df.unit.values
    test_df = explainer.get_predictions_as_df(x_test, y_test, lstm_pred_test)
    test_df["unit"] = test_engines

    def get_counter_and_factuals(train_cluster,
                                 train_cluster_df,
                                 selected_sample):

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
                                                ].predict(x_sample[np.newaxis,:]
                                                          )[0]

        closest_unhealthy_cluster = train_cluster["unhealthy_clusters"
                                               ]["kmeans"
                                                ].predict(x_sample[np.newaxis,:]
                                                          )[0]

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

        return factual_cluster, counterfactual_cluster, int(x_prediction)

    def plot_example_cluster(ax, cluster,
                             x_sample, is_unheathy,
                             is_counterfactual, x_lim):

        center_color = "green"
        sample_color = "green"
        if is_unheathy and is_counterfactual:
            center_color = "green"
            sample_color = "red"

        elif not is_unheathy and is_counterfactual:
            center_color = "red"

        elif is_unheathy and not is_counterfactual:
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

        # now plot your selected sample
        ax.scatter(x_lim, x_sample, s=60, c=sample_color, alpha=0.25,
                   label="predicted_sample")
        ax.plot(x_lim, x_sample, "--", c="black")


    dev_data = load_data(path, dev_mode)
    dev_data = cmapss.get_univariate_cmapss(dev_data, "s12")
    trainset, testset = cmapss.train_test_split(dev_data)
    testset = cmapss.minmax_scale(testset)
    testset = cmapss.denoise_sensors(testset)

    # ask the user which engine to plot
    test_engines = ["NONE"]+list(test_df.unit.unique())

    choose_engine = st.sidebar.selectbox("Choose an engine",
                                        test_engines)

    def plot_example_cluster(ax, cluster, c_label, x_sample,
                            is_unhealthy, is_counterfactual,
                           x_lim):    
        
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


        test_df = pd.read_csv("../results/test_df.csv", sep=',')
        test_engines = test_df.unit.values
        test_df = explainer.get_predictions_as_df(x_test, y_test, lstm_pred_test)
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
        plt.savefig("../results/e{}_y_true.png".format(engine_num))


        #################### now the explanation plot ####################
        fig, axs = plt.subplots(2, sharex=True, figsize=(20,10))
        fig.suptitle("Engine {} Predicted Labels".format(engine_num), 
                 fontsize=20)
    
        is_first_round = True
        for i in range(len(engine_data)):
        
            if i%2==1: continue
        
            print("=", end="")
            selected_sample = engine_data.loc[i]

            x_sample = list(engine_data.loc[i][:-4])
            factuals, counterfactuals, labels, x_pred = explainer.get_counter_and_factuals(train_cluster, train_cluster_df, selected_sample)

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
        plt.savefig("../results/e{}_explanation.png".format(engine_num))

    if(choose_engine != "NONE"):

        engine_explainer_parameters = {"train_cluster": train_cluster,
                                       "train_cluster_df": train_cluster_df,
                                       "x_test": x_test,
                                       "y_test": y_test,
                                       "lstm_pred_test": lstm_pred_test}


        # function
        explain_engine(engine_explainer_parameters, choose_engine)


        st.write('Explaining the prediction with factual and counterfactuals:')
        image_file = "../results/e{}_y_true.png".format(choose_engine)
        st.image(image_file)

        #image_file = "../results/e{}_y_pred.png".format(choose_engine)
        #st.image(image_file)

        image_file = "../results/e{}_explanation.png".format(choose_engine)
        st.image(image_file)


        def plot_local_counter_and_factuals(train_cluster, train_cluster_df, selected_sample, engine_num, seq_num):
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
    
            from sklearn import tree
    
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
    
    
            """ training the local Tree """
    
            # prepare the data
            dev_mode = True 
            dev_data = cmapss.load_data(path)
            dev_data = cmapss.cluster_operational_settings(dev_data,
                                                           dev_mode)
            dev_data = cmapss.calculate_TTF(dev_data)
            dev_data = cmapss.calculate_continues_healthstate(dev_data)
            dev_data = cmapss.calculate_descrete_healthstate(dev_data)
            
            from numpy import loadtxt

            x_train = loadtxt('../results/x_train.csv', delimiter=',')
            y_train = loadtxt('../results/y_train.csv', delimiter=',')
            x_test = loadtxt('../results/x_test.csv', delimiter=',')
            y_test = loadtxt('../results/y_test.csv', delimiter=',')
    
            # extract train sets time series features
            dev_features = explainer.extract_time_series_features(dev_data,"s12", x_train.shape[1])
    
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
            nn_model = load_lstm("lstm")
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
    
            # look at the decision cuts for thess clusters
            #display(explainer.tree_to_code(local_dt, feature_names))
            local_decisions, local_imp_features = tree_to_code(local_dt, feature_names)
    
            # get the features of the selected sample
            features = separate_test_features(selected_sample.name, feature_names, ts_features_test)
    
            def plot_example_cluster(cluster, x_sample, is_unhealthy, is_counterfactual, engine_num, seq_num):
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
                    center_color = "red" 
                    label="counterfactual"

                elif not is_unhealthy and is_counterfactual:
                    center_color = "red"
                    label="counterfactual"

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
                ax.set_yticks(np.arange(0.6, 1, 0.1))
                ax.legend(loc="lower right", fontsize=16)

                # now add the important timesteps
                for f in local_imp_features:
                    timestep = int(f.split("_")[1])
                    ax.axvspan(xmin=timestep-0.25,
                               xmax=timestep+0.25,
                               ymax=1,
                               facecolor='green', alpha=0.2)

        #     if x_model_confidence<50:
        #         x_model_confidence = -x_model_confidence + 100
        #     else:
    
            fig = plt.figure(figsize=(15,5))
            fig.suptitle(
                "y_true={}  | y_pred={}  |  sigmoid value={}\n\n\n\n\n".format(x_actual_label,
                                                               x_prediction,
                                                               np.round(
                                                                   x_model_confidence,2
                                                                   )),
                         fontsize=20)


            ax = plt.subplot(1, 2, 1)
            ax.set_title("Factual_Cluster vs Sample",fontsize=16)
            plot_example_cluster(factual_cluster, x_sample, x_prediction, False, choose_engine, seq_num)

            ax = plt.subplot(1, 2, 2)
            ax.set_title("Counterfactual_Cluster vs Sample", fontsize=16)
            plot_example_cluster(counterfactual_cluster, x_sample, x_prediction, True, choose_engine, seq_num)

            plt.savefig("../results/e{}_{}_explanation.png".format(engine_num, seq_num))

            plt.show()
    

            # now plot the tree features
            #local_decisions, local_imp_features = tree_to_code(local_dt, feature_names)
    
            fig = plt.figure(figsize=(15,16))
            ax = plt.subplot(3, 1, 1)#i+3)
            ax.set_title("Decision Cuts", fontsize=20)
            ax.grid(False)
    
            local_decisions = [
                "A sequence is healthy when:",
                "    mean_0 <= 0.7350 & mean_19 > 0.7072 & fft_5 <= 0.001",
                "A sequence becomes unhealthy when fft_5 > 0.0011",
                "",
                "A sequence is healthy when:",
                "    mean_0 > 0.7350 & fft_1 <= 0.0121 & mean_5 <= 0.7386",
                "A sequence becomes unhealthy when mean_5 > 0.7386",
                "",
                "A sequence is healthy when:",
                "    mean_0 > 0.7350 & fft_1 > 0.0121 & std_0 > 0.2782 & fft_3 <= 0.0075",
                "A sequence becomes unhealthy when fft_3 > 0.0075" 
            ]
    
            for j in range(len(local_decisions)):
                ax.text(0.03, j*-0.08 + 0.9, local_decisions[j], fontsize = 20)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            #plt.tight_layout()
            plt.savefig("../results/e{}_{}_rule_explanation.png".format(engine_num, seq_num))


    
        def tree_to_code(tree, feature_names):
    
            from sklearn.tree import _tree

            tree_ = tree.tree_
            feature_name = [
                feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
                for i in tree_.feature
            ]

            def recurse(node, depth, decision_cuts, informative_features):
                indent = "  " * depth
                if tree_.feature[node] != _tree.TREE_UNDEFINED:
                    name = feature_name[node]
                    threshold = tree_.threshold[node]
                    print("{}if {} <= {}:".format(indent, name, threshold))
                    decision_cuts.append("{}if {} <= {}:".format(
                        indent, name, np.round(threshold, 4)))
                    informative_features.append(name)
                    recurse(tree_.children_left[node], depth + 1, decision_cuts, informative_features)
                    print("{}else:  # if {} > {}".format(indent, name, threshold))
                    decision_cuts.append("{}else:  # if {} > {}".format(
                        indent, name, np.round(threshold, 4)))
                    informative_features.append(name)
                    recurse(tree_.children_right[node], depth + 1, decision_cuts, informative_features)
                else:
                    print("{}return {}".format(indent, np.argmax(tree_.value[node][0])))
                    decision_cuts.append("{}return {}".format(indent, np.argmax(tree_.value[node][0])))

            local_decision_cuts = [] 
            influencial_features = []
            recurse(0, 1, local_decision_cuts, influencial_features)
            return local_decision_cuts, list(set(influencial_features))


        def separate_test_features(index, feature_names, ts_features_test):
            features = {}
            values=[]
            for i in range(len(feature_names)):
                name = feature_names[i].split("_")[0]
                values.append(np.round(ts_features_test[index][i], 4))

                #else: # there is a new name, save what you had and reset the cash
                if len(values)-20==0: # harcoded (window-size)
                    features.update({name:values})
                    values=[]
            
            del features['mean'] # drop the actual sequence
            return features

        
        unit_samples = test_df[test_df.unit==choose_engine].index

        # which sample you would like to explain?
        choose_window = st.sidebar.selectbox("Choose a time window (a sequence)",
                                    range(1, len(unit_samples)-1))
        selected_sample = test_df.iloc[unit_samples[choose_window]]

        plot_local_counter_and_factuals(train_cluster, train_cluster_df, selected_sample, choose_engine, choose_window)

        st.write('Explaining the time window with factual and counterfactuals, and decision cuts:')
        image_file = "../results/e{}_{}_explanation.png".format(choose_engine, choose_window)
        st.image(image_file)

        image_file = "../results/e{}_{}_rule_explanation.png".format(choose_engine, choose_window)
        st.image(image_file)