# MNIST---PROJECT
MNIST Multi-Model Classification – ML Mini Project

This project builds, evaluates, and compares multiple machine learning classifiers on the MNIST digit recognition dataset.
It also identifies the best model (SVM) using ROC–AUC analysis and further improves it using GridSearchCV hyperparameter tuning.

Project Features
✔ Load & preprocess MNIST dataset
✔ Train 9 ML models:

KNN

Naive Bayes

Logistic Regression

Decision Tree

Random Forest

AdaBoost

Gradient Boosting

XGBoost

SVM (RBF Kernel)

✔ Compute:

Accuracy

Confusion Matrix

Classification Report

ROC Curve (Micro-Average)

ROC-AUC score for each model

✔ Identify Best Model Using AUC
✔ Tune Best Model (SVM) using GridSearchCV
✔ Retrain SVM using best parameters
✔ Final ROC curve plot and final accuracy
ML Mini Project-1/
│
├── MNIST.py               # Main project code
├── mnist_train.csv        # Training dataset
├── requirements.txt       # Required packages
├── python_copy/           # Your virtual environment folder
└── README.md              # Project documentation
