# Aurora_AI
Artificial Inteligence created at NASA´s Hackathon, to predict solar tempests.

Tho access and use the data, you can acess our database present here:
https://drive.google.com/drive/folders/1Jrlf73-H-vAi_TUq4fdjE6nvBW-O4639?usp=sharing

Or, you can access NASA database and download the "raw" data about DSCOVR in this link:
https://www.spaceappschallenge.org/develop-the-oracle-of-dscovr-experimental-data-repository/

# Geomagnetic Kp Index Prediction using Random Forest Regressor
This repository contains code for training and evaluating a machine learning model to predict the geomagnetic Kp index. The model is trained using the LightGBM algorithm.

Repository Contents
aurora_ia.py: This file contains code for:
Data loading and preprocessing
Splitting the data into training and test sets
Model training using the LightGBM algorithm
Evaluation of the trained model using RMSE and a custom accuracy metric
Saving the trained model
aurora_test_code.py: This file contains code for:
Data loading and preprocessing
Predicting using a previously saved LightGBM model
Evaluating the model's performance on new data
Setup
Prerequisites
Ensure you have the following libraries installed:

lightgbm
pandas
sklearn
Usage
Train the model:
bash
Copy code
$ python aurora_ia.py
This will train the model on the provided data and save the trained model as lgbm_model.txt.

Evaluate the model:
bash
Copy code
$ python aurora_test_code.py
This will evaluate the model's performance on new data and print the model's accuracy.

Overview
The geomagnetic Kp index is crucial for understanding space weather phenomena. Accurate predictions of this index can help in mitigating the adverse effects of space weather on technology and infrastructure.

The model in this repository uses the LightGBM algorithm, which is a gradient boosting framework that uses tree-based algorithms. This choice was made based on its high accuracy and robustness against overfitting.

Acknowledgments
This work was inspired by the need for accurate and real-time space weather predictions. We would like to thank the entire team for their contributions and the open-source community for the tools that made this project possible.
