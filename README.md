# eXplainable Deep Neural Networks for Machine's Health Prognosis Management applications

This repository contains scripts and reports on explaining Deep Learning applications for Remaining Useful Life and Machine's health Estimation.

**Project Type** - Bring your own method

**Project Motivation** - This project is part of a PhD in progress and is motivated by the work of 
* [Ribeiro et. al., 2016 "Why should I trust you?" Explaining the predictions of any classifier. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining](https://www.kdd.org/kdd2016/papers/files/rfp0573-ribeiroA.pdf)
* [Schmidt et al., 2019. Quantifying interpretability and trust in machine learning systems. AAAI-19 Workshop on Network Interpretability for Deep Learning](https://arxiv.org/abs/1901.08558)
* [Guilleme et al., 2019. Agnostic local explanation for time series classification. 2019 IEEE 31st International Conference on Tools with Artificial Intelligence (ICTAI)](https://ieeexplore.ieee.org/document/8995349/)
* [Guidotti et al., 2018. Local rule-based explanations ofblack box decision systems. arXiv preprint arXiv:1805.10820.](https://arxiv.org/abs/1805.10820)

## A Road Map to the Repository
This repository contains the following Python (3.8) scripts.

1. __prepare_dara.py__: this script contains the class CMAPSS and is meant to preprocess the CMAPSS development dataset.
2. __inspect_results.py__: this script comprises functions that plot the model predictions and compare them with their actual labels. 
3. __explainability.py__: this script contains the functions I used to generate the factual and counterfactual examples. It also contains the method that extracts the surrogate model's time series features.

The __CMAPSS_main.ipynb__ contains the experiments and documentation I created to model the CMAPSS dataset and predict the health state (healthy or unhealthy) of a given engines' sequence.

As this is a binary classification task, I evaluate my model's performance by its accuracy and f1-score. I aim to achieve ~85% accuracy and ~80% f1-score. However, until now and after parameter tuning, my model reaches an accuracy of 79% and an average f1-score of 79%, having 76% and 82% f1-score for class healthy and unhealthy, respectively.

### A rough estimation of my workload
I have spent roughly 86 hours implementing the code, debugging, parameter tuning, refactoring, and cleaning up the scripts. 
Unfortunately, after transferring my code into the scripts, I ran into some annoying bugs, which took me some good hours to fix.
The breakdown of my hours are as the following:

- WP 1: ~24 hours
-- Data preprocessing and feature engineering > (~20 hours)
-- Reading papers for a benchmark close to my task > (~4 hours)
- WP 2:  ~31 hour
-- Model building / baselines/vanilla LSTM > (~2 hours)
-- Parameter tunning (Before Talos Library) > (~20 hours)
-- Parameter tunning (After Talos Library) > (~1 hour)
-- Model Inspection > (~8 hours)
- WP 3: ~31 hour
-- Factual and Counterfactual explanations > (~30 hours)
-- surrogate Model > (~1 hour)
- WP 4:  
- -- As I did it simultaneously while experimenting, I can only guess around 10 hours to create my January slides and an overall image. But these hours are also calculated within each of the hours mentioned in the above work packages. 



## Project Description and Summary
Interpretable machine learning has recently attracted a lot of interest in the community. The current explainability approaches mainly focus on models trained on non-time series data. LIME and SHAP are well-known post-hoc examples that provide visual explanations of feature contributions to model decisions on an instance basis. Other approaches, such as attribute-wise interpretations, only focus on tabular data. Little research has been done so far on the interpretability of predictive models trained on time series data. Therefore, my Ph.D. focuses on explaining decisions made by black-box models such as Deep Neural Networks trained on sensor data. 
In my [publication](https://papers.phmsociety.org/index.php/phme/article/view/1244), I first presented the results of a qualitative study, in which we systematically compared the types of explanations and the properties (e.g., method, computational complexity) of existing interpretability approaches for models trained on the PHM08-CMAPSS dataset. In our subsequent work, we investigated machine learning practitioners' needs to advance and improve the explanations in terms of comprehensiveness, comprehensibility, and trust. We also pointed out the advantages and disadvantages of using these approaches for interpreting models trained on time series data.

For this project, I extend the idea of LIME and LORE to generate explanations for a binary time series classifier. 
Lime uses a local surrogate interpretable model (e.g., Lasso or a Decision Tree) to estimate the black-box's decision boundary in local neighborhoods. For this purpose, LIME first generates extra samples and their corresponding labels (predicted by the black-box) around a data point of interest. Second, based on their distance to the data point, LIME weights the generated sample - the closest to the data point, the higher the weight. Finally, LIME justifies the black-box's prediction using the trained weights of an interpretable surrogate model on this generated dataset. However, these explanations are not completely meaningful for sensor data (e.g., vibration data). 

Therefore, I propose to generate the new dataset by generating interpretable representations such as time domain (e.g., amplitute mean, pitch, std) and frequency domain features (e.g., ffts) from the raw input time series. 
Then, using a surrogate decision tree, we can extract the boundaries of the feature parameters, and use them as explanations.   
These explanations point to relevant time series characteristics and their parameters, representing their contribution to the classifier's decision. 

Furthermore, I extract factual and counterfactual examples that visually can show the differences between the time series sequences from the prediction set and the train set.

To evaluate my idea, I use the PHM08-CMAPSS dataset, a well-known and publicly available run-to-failure sensor data. The figure below shows an overview of my approach for extracting similar examples and counterfactual examples for a given prediction.


![alt text][BigPicture]

[BigPicture]: figures/Big-Picture.png


### Dataset and Work Description
Commercial Modular Aero-Propulsion System Simulation is intended for Prognosis Health Management tasks such as Time-To-Failure (TTF) and Remaining-Useful-Life (RUL) estimation. We specifically downloaded the publicly available CMAPSS dataset used for the PHM08 challenge from the [Nasa dataset repository](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/). C-MAPSS aircraft engine data is a benchmark dataset that contains run-to-failure data, including labeled breakdowns. It represents 218 engines of a similar type, which all start from a healthy state. Faults are injected throughout the entire engine's life span until it goes to a breakdown state. The maximum and the minimum number of cycles to failure in the training set are 357 and 128, respectively, with a mean of 210. The engine data's attributes (26 attributes) consist of three operational settings, time series data collected from vibration sensors. Given the original description of the dataset at [Nasa dataset repository](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/), we name the columns as follows: the first two columns represent the engine number and cycles, columns 3 to 5 represent the operational settings, and from column six, every time series attribute is named as s1 to s21 (sensor-measurement one, sensor-measurement two and so on). For this project, I focus on a univariate time series classification task and choose the sensor measurement with the strongest influence on classification results. The selected sensor measurement is based on my previous [experiments](https://papers.phmsociety.org/index.php/phme/article/view/1244), which showed that sensor-measurement 12 (s12) has the highest contribution to the linear models' decision.

![alt text][cmapss]

[cmapss]: figures/CMAPSS_description.png

### Project Timeline 

![alt text][wps]

[wps]: figures/WPs.png

![alt text][timeline]

[timeline]: figures/timeline.png


