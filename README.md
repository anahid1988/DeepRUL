# eXplainable Deep Neural Networks for Machine's Health Prognosis Management applications

This repository will contain experiments on explaining Deep Learning applications for Remaining Useful Life and Machine's health Estimation.

**Project Type** - Bring your own method

**Project Motivation** - This project is a PhD in progress and is motivated by the work of 
* [Ribeiro et. al., 2016 "Why should I trust you?" Explaining the predictions of any classifier. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining](https://www.kdd.org/kdd2016/papers/files/rfp0573-ribeiroA.pdf)
* [Schmidt et al., 2019. Quantifying interpretability and trust in machine learning systems. AAAI-19 Workshop on Network Interpretability for Deep Learning](https://arxiv.org/abs/1901.08558)
* [Guilleme et al., 2019. Agnostic local explanation for time series classification. 2019 IEEE 31st International Conference on Tools with Artificial Intelligence (ICTAI)](https://ieeexplore.ieee.org/document/8995349/)
* [Guidotti et al., 2018. Local rule-based explanations ofblack box decision systems. arXiv preprint arXiv:1805.10820.](https://arxiv.org/abs/1805.10820)


## Project Description and Summary
Interpretable machine learning has recently attracted a lot of interest in the community. Currently, it mainly focuses on models trained on non-time series data. LIME and SHAP are well-known examples and provide visual explanations of feature contributions to model decisions on an instance basis. Other post-hoc approaches, such as attribute-wise interpretations, also focus on tabular data only. Little research has been done so far on the interpretability of predictive models trained on time series data. Therefore, this work focuses on explaining decisions made by black-box models such as Deep Neural Networks trained on sensor data. In this project, we first present the results of a qualitative study, in which we systematically compare the types of explanations and the properties (e.g., method, computational complexity) of existing interpretability approaches for models trained on the PHM08-CMAPSS dataset. We further investigate machine learning practitioners needs to advance and improve the explanations in terms of their comprehensiveness, comprehensibility and trust. Finally, we expand the idea of LIME for meaningful time series model interpretation and evaluate our approach on PHM08-CMAPSS dataset. Throughout our experiments, we also point out the advantages and disadvantages of using these approaches for interpreting models trained on time series data. Our investigation results can serve as a guideline for selecting a suitable explainability method for black-box predictive models trained on time-series data.

![alt text][cmapss]

[cmapss]: https://github.com/anahid1988/DeepRUL/blob/master/figures/CMAPSS_description.png

### Dataset Description
Commercial Modular Aero-Propulsion System Simulation is intended for Prognosis Health Management tasks such as Time-To-Failure (TTF) and Remaining-Useful-Life (RUL) estimation. We specifically download the publicly available CMAPSS dataset used for PHM08 challenge from [Nasa dataset repository](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/). 
C-MAPSS aircraft engine data is a benchmark dataset that contains run-to-failure data, including labeled breakdowns. It represents 218 engines of a similar type, which all start from a healthy state. Faults are injected throughout the entire engine's life span until it goes to a breakdown state. The maximum and the minimum number of cycles to failure in the training set are 357 and 128, respectively, with a mean of 210. The engine data's attributes consist of three operational settings, vibration data collected from vibration sensors, and two binary attributes. We name the columns as follows: the first two columns represent the engine number and cycles, columns 3 to 5 represent the operational settings and from column six, and every attribute is named as s1 to s21 (sensor 1, sensor two and so on). 

### Project Timeline

![alt text][timeline]

[timeline]: https://github.com/anahid1988/DeepRUL/blob/master/figures/project_timeline.png
