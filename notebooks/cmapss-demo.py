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

import warnings
warnings.filterwarnings("ignore")
np.random.seed(7)


# Title of the Demo
st.title("""eXplainable Deep Neural Networks for Machine Health Prognosis
applications""")

st.sidebar.title("eXplainable PdM")
st.sidebar.markdown("Is the turbofan engine healthy or unhealthy?")

if st.sidebar.checkbox("Display the description", False):
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


# parameters
path = "../../Datasets/PHM08_Challenge_Data/train.txt"
dev_mode = True # not using the CMAPSS test set

#@st.cash
cmapss = prepare_data.CMAPSS()

@st.cache
def load_data(p, mode):
    data = cmapss.load_data(path)
    data = cmapss.cluster_operational_settings(data, mode)
    data = cmapss.calculate_TTF(data)
    data = cmapss.calculate_continues_healthstate(data)
    data = cmapss.calculate_descrete_healthstate(data)

    return data

dev_data = load_data(path, dev_mode)
dev_data = cmapss.get_univariate_cmapss(dev_data, "s12")
x_train, x_test = cmapss.train_test_split(dev_data)
x_train = cmapss.minmax_scale(x_train)
x_train = cmapss.denoise_sensors(x_train)

x_test = cmapss.minmax_scale(x_test)
x_test = cmapss.denoise_sensors(x_test)

# display DataFrame
if st.sidebar.checkbox("Display CMAPSS data", False):
    st.subheader('CMAPSS Data')
    st.write('Data Source: https://data.nasa.gov/widgets/xaut-bemq')
    st.write("""Commercial Modular Aero-Propulsion System Simulation is intended
     for Prognosis Health Management tasks such as Time-To-Failure (TTF) and
     Remaining-Useful-Life (RUL) estimation.""")
    st.write(dev_data)

    st.write('Calculated Engine Health Status:')
    image_file = "./results/figures/true_healthstatus.png"
    st.image(image_file)

    st.write('Speed Fan (raw data):')
    fig, ax = plt.subplots()
    df = dev_data[dev_data.unit==1]
    ax.set_title("Engine 1 - Speed Fan over ~260 flights")
    df.s12.plot(figsize=(10, 3), color="gray")
    st.write(fig)

    st.write("Engine 1 - Normalized and denoised Speed Fan  over ~260 flights")
    fig, ax = plt.subplots()
    df = x_train[x_train.unit==1]
    ax.set_title("Engine 1 - Speed Fan over ~260 flights")
    df.s12.plot(figsize=(10, 3), color="gray")
    ax.set_yticks(np.arange(0,1,0.1))
    st.write(fig)

x_train = np.loadtxt('./results/x_train.csv', delimiter=',')
y_train = np.loadtxt('./results/y_train.csv', delimiter=',')
x_test = np.loadtxt('./results/x_test.csv', delimiter=',')
y_test = np.loadtxt('./results/y_test.csv', delimiter=',')

if st.sidebar.checkbox("Display Baseline Reports", False):

    st.subheader("Baseline Report")

    @st.cache
    def report_baseline(model_name, x, y):
        with open('./results/{}.pkl'.format(model_name), 'rb') as f:
            sk_model = pickle.load(f)
        f.close()

        y_hat = sk_model.predict(x)
        cm = confusion_matrix(y, y_hat,
                              labels=[0, 1])
        return y_hat, cm


    choose_model = st.sidebar.selectbox("Choose the Baseline Model",
    		["NONE","LogisticRegressionCV", "RidgeClassifierCV",
            "KNeighborsClassifier", "DecisionTreeClassifier"])

    if(choose_model != "NONE"):
        y_pred, cf_matrix = report_baseline(choose_model, x_test, y_test)

        st.write('\n{} Classification Report:\n'.format(choose_model))

        class_names=["healthy", "unhealthy"]
        st.write("Accuracy: {}%".format(accuracy_score(y_test, y_pred
        ).round(4)*100))
        st.write("Precision: {}%".format(precision_score(y_test, y_pred,
        labels=class_names).round(4)*100))

        st.write("Recall: {}%".format(
            (recall_score(y_test, y_pred, labels=class_names).round(4)*100
            ).round(2)))
        st.write("F1 Score: {}%".format(f1_score(
                    y_test, y_pred, labels=class_names).round(4)*100))

        st.subheader('Confusion Matrix:')
        st.write(cf_matrix)
        fig, ax = plt.subplots(figsize=(3,2))
        sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True,
                fmt='.2%', cmap='Blues')
        st.pyplot(fig)

nn_model = None

if st.sidebar.checkbox("Display LSTM Reports", False):
    #@st.cache
    def load_lstm(name):
        model = keras.models.load_model("./results/"+name)
        return model

    nn_model = load_lstm("LSTM")

    lstm_pred_test = nn_model.predict(x_test[:, :, np.newaxis])
    cm = confusion_matrix(y_test,
                          inspect.continues_to_binary(lstm_pred_test),
                          labels=[0, 1])

    st.write('Model Summary:')
    image_file = "./results/figures/LSTM_Summary.png"
    st.image(image_file)

    st.write('LSTM Classification Report:\n')

    class_names=["healthy", "unhealthy"]
    st.write("Accuracy: {}%".format(accuracy_score(y_test,
    inspect.continues_to_binary(lstm_pred_test)
    ).round(4)*100))
    st.write("Precision: {}%".format(precision_score(y_test,
    inspect.continues_to_binary(lstm_pred_test),
    labels=class_names).round(4)*100))

    st.write("Recall: {}%".format(
        (recall_score(y_test,
        inspect.continues_to_binary(lstm_pred_test),
         labels=class_names).round(4)*100
        ).round(2)))
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


@st.cache
def load_cluster_models(file_name):
    cluster_df = pd.read_csv("./results/{}_df.csv".format(file_name), sep=',')

    with open('./results/{}.pkl'.format(file_name), 'rb') as f:
        cluster = pickle.load(f)
    f.close()

    return cluster_df, cluster
train_cluster_df, train_cluster = load_cluster_models("train_cluster")



if st.sidebar.checkbox("Globaly Explain LSTM", False):

    if not nn_model:
        def load_lstm(name):
            model = keras.models.load_model(name)
            return model

        nn_model = load_lstm("./results/LSTM")

        lstm_pred_test = nn_model.predict(x_test[:, :, np.newaxis])

    if st.sidebar.checkbox("Surrogate Tree", False):
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
        st.write("Accuracy: {}%".format(accuracy_score(
         inspect.continues_to_binary(lstm_pred_test),
         surrogate_dt_test_preds
         ).round(4)*100))
        st.write("Precision: {}%".format(precision_score(
         inspect.continues_to_binary(lstm_pred_test),
         surrogate_dt_test_preds,
         labels=class_names).round(4)*100))

        st.write("Recall: {}%".format(
            (recall_score(
            inspect.continues_to_binary(lstm_pred_test),
            surrogate_dt_test_preds, labels=class_names).round(4)*100
            ).round(2)))
        st.write("F1 Score: {}%".format(f1_score(
                    inspect.continues_to_binary(lstm_pred_test),
                    surrogate_dt_test_preds,
                    labels=class_names).round(4)*100))

        st.subheader('Confusion Matrix:')
        sdt_cm = confusion_matrix(inspect.continues_to_binary(lstm_pred_test),
                              surrogate_dt_test_preds,
                              labels=[0, 1])

        st.write(sdt_cm)
        fig, ax = plt.subplots(figsize=(3,2))
        sns.heatmap(sdt_cm/np.sum(sdt_cm), annot=True,
                fmt='.2%', cmap='Blues')
        st.pyplot(fig)

        st.write("Most Influential Features:")
        importance = surrogate_dt.feature_importances_
        # summarize feature importance
        f_imp = pd.DataFrame(importance, columns=["importance"])
        f_imp.index = feature_names
        f_imp["importance"] = f_imp.importance.values * 100

        f_imp=f_imp.sort_values(by=['importance'], ascending=False)
        st.write(f_imp.head(10))

        # plot feature importance
        st.bar_chart(f_imp[:10])

        st.write('Surrogate Tree Decision Rules:')
        image_file = "./results/figures/surrogate_tree.png"
        st.image(image_file)
    if st.sidebar.checkbox("Model Decision Boundary", False):
        st.write("""All the predictions:""")
        inspect.visualize_predictions(train_cluster_df, "LSTM")
        image_file = "./results/decision_boundary.png"
        st.image(image_file)

        st.write("""Failed Predictions:""")
        inspect.visualize_predictions(train_cluster_df[
        train_cluster_df.y_true != train_cluster_df.y_pred],
                      "LSTM")
        image_file = "./results/decision_boundary.png"
        st.image(image_file)

        st.write("""False Positives:""")
        inspect.visualize_predictions(train_cluster_df[
        (train_cluster_df.y_true==0) & (train_cluster_df.y_pred==1)],
                      "LSTM")
        image_file = "./results/decision_boundary.png"
        st.image(image_file)

        st.write("""False Negatives:""")
        inspect.visualize_predictions(train_cluster_df[
        (train_cluster_df.y_true==1) & (train_cluster_df.y_pred==0)],
                      "LSTM")
        image_file = "./results/decision_boundary.png"
        st.image(image_file)

if st.sidebar.checkbox("Locally Explain LSTM", False):
    if not nn_model:
        def load_lstm(name):
            model = keras.models.load_model(name)
            return model

        nn_model = load_lstm("./results/LSTM")

        lstm_pred_test = nn_model.predict(x_test[:, :, np.newaxis])

    test_df = pd.read_csv("test_df.csv", sep=',')
    test_engines = test_df.unit.values
    test_df = explainer.get_predictions_as_df(x_test, y_test, lstm_pred_test)
    test_df["unit"] = test_engines

    @st.cache
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


    trainset, testset = cmapss.train_test_split(dev_data)
    testset = cmapss.minmax_scale(testset)
    testset = cmapss.denoise_sensors(testset)

    # ask the user which engine to plot
    test_engines = ["NONE"]+list(test_df.unit.unique())

    choose_engine = st.sidebar.selectbox("Choose an engine",
                                        test_engines)


    def plot_example_cluster(ax, cluster, x_sample, is_unheathy,
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

    @st.cache
    def explain_engine(engine_num):
        x_engine=[]
        conf_engine=[]
        y_engine = []
        y_hat_engine = []

        #-1 = unit #-2 = conf #-3 = pred # -4 = true
        for i in range(len(test_df[test_df.unit==engine_num])):
            engine_data = test_df[test_df.unit==engine_num].reset_index(drop=True)
            x_engine += list(engine_data.loc[i][:-4])

            conf_engine += [engine_data.loc[i][-2] for _ in range(20)]
            y_hat_engine += [engine_data.loc[i][-3] for _ in range(20)]
            y_engine += [engine_data.loc[i][-4] for _ in range(20)]

        plt.figure(figsize=(20,5))
        plt.title("Engine {} Actual Labels".format(engine_num),
                 fontsize=20)
        plt.plot(x_engine, c="black", label="Engine {}".format(engine_num))
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


        plt.figure(figsize=(20,5))
        plt.title("Engine {} Predicted Labels".format(engine_num),
                 fontsize=20)
        plt.plot(x_engine, c="black", label="Engine {}".format(engine_num))
        for i in range(len(y_hat_engine)):
            if y_hat_engine[i]==0:
                plt.axvspan(xmin=i, xmax=i+1, ymax=1,
                            facecolor='green', alpha=0.2)
            else:
                plt.axvspan(xmin=i, xmax=i+1, ymax=1,
                            facecolor='red', alpha=0.2)
        plt.ylim(0,1.1,0.2)
        plt.legend(fontsize=14)
        plt.savefig("./results/e{}_y_pred.png".format(engine_num))

        plt.plot(conf_engine, c="gray", label="Sigmoid")

        fig, axs = plt.subplots(2, sharex=True, figsize=(20,10))

        is_first_round = True
        for i in range(len(engine_data)):
            print("=", end="")
            selected_sample = engine_data.loc[i]

            x_sample = list(engine_data.loc[i][:-4])
            factuals, counterfactuals, x_pred = get_counter_and_factuals(
             train_cluster, train_cluster_df, selected_sample)

            if is_first_round:
                x_axis_loc = 0
                # as long as the sequence length
                vertical_lim = [vl for vl in range(20)]
            else:
                x_axis_loc +=20
                vertical_lim = [x_axis_loc+vl for vl in range(20)]

            plot_example_cluster(axs[0], factuals, x_sample, x_pred, False, vertical_lim)
            plot_example_cluster(axs[1], counterfactuals, x_sample, x_pred, True, vertical_lim)

            is_first_round=False

        axs[0].set_title("Factual Examples", fontsize=20)
        axs[0].set_xticks(np.arange(0, max(vertical_lim), 20))
        axs[0].set_yticks(np.arange(0, 1.1, 0.1))
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        axs[1].set_title("CounterFactual Examples", fontsize=20)
        axs[1].set_xticks(np.arange(0, max(vertical_lim), 20))
        axs[1].set_yticks(np.arange(0, 1.1, 0.1))
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        plt.show()
        plt.savefig("./results/e{}_explanation.png".format(engine_num))

    if(choose_engine != "NONE"):
        # function
        explain_engine(choose_engine)

        st.write('Explaining the prediction with factual and counterfactuals:')
        image_file = "./results/e{}_y_true.png".format(choose_engine)
        st.image(image_file)

        image_file = "./results/e{}_y_pred.png".format(choose_engine)
        st.image(image_file)

        image_file = "./results/e{}_explanation.png".format(choose_engine)
        st.image(image_file)
