# eXplainable Deep Neural Networks for Machine's Health Prognosis Management applications

This repository will contain reports on explaining Deep Learning applications for Remaining Useful Life and Machine's health Estimation.

**Project Type** - Bring your own method

**Project Motivation** - This project is a PhD in progress and is motivated by the work of 
* [Ribeiro et. al., 2016 "Why should I trust you?" Explaining the predictions of any classifier. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining](https://www.kdd.org/kdd2016/papers/files/rfp0573-ribeiroA.pdf)
* [Schmidt et al., 2019. Quantifying interpretability and trust in machine learning systems. AAAI-19 Workshop on Network Interpretability for Deep Learning](https://arxiv.org/abs/1901.08558)
* [Guilleme et al., 2019. Agnostic local explanation for time series classification. 2019 IEEE 31st International Conference on Tools with Artificial Intelligence (ICTAI)](https://ieeexplore.ieee.org/document/8995349/)
* [Guidotti et al., 2018. Local rule-based explanations ofblack box decision systems. arXiv preprint arXiv:1805.10820.](https://arxiv.org/abs/1805.10820)


## Project Description and Summary
Interpretable machine learning has recently attracted a lot of interest in the community. The current explainability approaches mainly focus on models trained on non-time series data. LIME and SHAP are well-known post-hoc examples that provide visual explanations of feature contributions to model decisions on an instance basis. Other approaches, such as attribute-wise interpretations, only focus on tabular data. Little research has been done so far on the interpretability of predictive models trained on time series data. Therefore, my Ph.D. focuses on explaining decisions made by black-box models such as Deep Neural Networks trained on sensor data. 
In my [publication](https://papers.phmsociety.org/index.php/phme/article/view/1244), I first presented the results of a qualitative study, in which we systematically compared the types of explanations and the properties (e.g., method, computational complexity) of existing interpretability approaches for models trained on the PHM08-CMAPSS dataset. In our subsequent work, we investigated machine learning practitioners' needs to advance and improve the explanations in terms of comprehensiveness, comprehensibility, and trust. 
We also pointed out the advantages and disadvantages of using these approaches for interpreting models trained on time series data.
For this project, I expand the idea of LIME for generating meaningful explanations on time series model outputs. Lime uses a local surrogate interpretable model (e.g., Lasso or a Decision Tree) to estimate the black-box's decision boundary in local neighborhoods. For this purpose, LIME first generates extra samples and their corresponding labels (predicted by the black-box) around a data point of interest. Second, based on their distance to the data point, LIME weights the generated sample - the closest to the data point, the higher the weight. Finally, LIME justifies the black-box's prediction using the trained weights of an interpretable surrogate model on this generated dataset.
However, these explanations are not completely meaningful for sensor data (e.g., vibration data). Therefore, I propose to generate the new dataset by generating interpretable representations for the time series. Then, using a local surrogate decision tree, we can extract the boundaries of the features and use them as explanations.   
These explanations point to relevant time series characteristics and their parameters (e.g., time domain and frequency domain features), representing their contribution to the classifier's decision. I use the PHM08-CMAPSS dataset to evaluate this idea, which is a well-known and publicly available run-to-failure sensor data.


![alt text][cmapss]

[cmapss]: figures/CMAPSS_description.png

### Dataset and Work Description
Commercial Modular Aero-Propulsion System Simulation is intended for Prognosis Health Management tasks such as Time-To-Failure (TTF) and Remaining-Useful-Life (RUL) estimation. We specifically download the publicly available CMAPSS dataset used for PHM08 challenge from [Nasa dataset repository](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/). 
C-MAPSS aircraft engine data is a benchmark dataset that contains run-to-failure data, including labeled breakdowns. It represents 218 engines of a similar type, which all start from a healthy state. Faults are injected throughout the entire engine's life span until it goes to a breakdown state. The maximum and the minimum number of cycles to failure in the training set are 357 and 128, respectively, with a mean of 210. The engine data's attributes (26 attributes) consist of three operational settings, time series data collected from vibration sensors. Given the original description of the dataset at [Nasa dataset repository](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/), we name the columns as follows: the first two columns represent the engine number and cycles, columns 3 to 5 represent the operational settings and from column six, every time series attribute is named as s1 to s21 (sensor-measurement one, sensor-measurement two and so on). 
For this project, I focus on a univariate time series classification task, and choose the sensor-measurement with the highest influence of classification results. The selected sensor-measurement is based on my previous [experiments](https://papers.phmsociety.org/index.php/phme/article/view/1244), which showed that the sensor-measurement 12 (s12) has the highest contribution to the linear models' decision. 

![alt text][wps]

[wps]: figures/WPs.png

### Project Timeline 
Figure below, shows an overview of my approach for extracting the similar examples and counterfactual examples for a given prediction.


![alt text][BigPicture]

[workpackages]: figures/Big-Picture.png

![alt text][timeline]

[timeline]: figures/timeline.png


