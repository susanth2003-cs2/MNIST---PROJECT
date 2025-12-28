import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

class MNIST():
    def __init__(self, path):
      try:
          self.df = pd.read_csv(path)
          self.df.fillna(0, inplace=True)
          self.df = self.df.astype(np.int16)
          self.X = self.df.iloc[:, 1:]
          self.y = self.df.iloc[:,0]
          self.classes = list(np.unique(self.y))
          self.y_bin = label_binarize(self.y, classes=self.classes)
          self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
          self.X, self.y, test_size=0.2, random_state=42 )
          self.y_test_bin = label_binarize(self.y_test, classes=self.classes)
          self.df_small = self.df.head(600)            # for GridSearch only
          self.X_small = self.df_small.iloc[:, 1:]
          self.y_small = self.df_small.iloc[:, 0]

      except Exception as e:
          ex_type, ex_msg, ex_line = sys.exc_info()
          print(f"Init Error at line {ex_line.tb_lineno}: {ex_msg}")

    def train_performance(self, model):
        print(f"Train Accuracy : {accuracy_score(self.y_train, model.predict(self.X_train))}")
        print(f"Train Confusion :\n{confusion_matrix(self.y_train, model.predict(self.X_train))}")
        print(f"Train Report :\n{classification_report(self.y_train, model.predict(self.X_train))}")

    def test_performance(self, model):
        print(f"Test Accuracy : {accuracy_score(self.y_test, model.predict(self.X_test))}")
        print(f"Test Confusion :\n{confusion_matrix(self.y_test, model.predict(self.X_test))}")
        print(f"Test Report :\n{classification_report(self.y_test, model.predict(self.X_test))}")

    def knn(self):
        self.knn_reg = KNeighborsClassifier()
        self.knn_reg.fit(self.X_train, self.y_train)
        self.train_performance(self.knn_reg)
        self.test_performance(self.knn_reg)

    def nb(self):
        self.nb_reg = GaussianNB()
        self.nb_reg.fit(self.X_train, self.y_train)
        self.train_performance(self.nb_reg)
        self.test_performance(self.nb_reg)

    def lr(self):
        self.lr_reg = LogisticRegression()
        self.lr_reg.fit(self.X_train, self.y_train)
        self.train_performance(self.lr_reg)
        self.test_performance(self.lr_reg)

    def dt(self):
        self.dt_reg = DecisionTreeClassifier()
        self.dt_reg.fit(self.X_train, self.y_train)
        self.train_performance(self.dt_reg)
        self.test_performance(self.dt_reg)

    def rf(self):
        self.rf_reg = RandomForestClassifier()
        self.rf_reg.fit(self.X_train, self.y_train)
        self.train_performance(self.rf_reg)
        self.test_performance(self.rf_reg)

    def ada(self):
        t = LogisticRegression()
        self.ada_reg = AdaBoostClassifier(estimator = t, n_estimators=10)
        self.ada_reg.fit(self.X_train, self.y_train)
        self.train_performance(self.ada_reg)
        self.test_performance(self.ada_reg)

    def gb(self):
        self.gb_reg = GradientBoostingClassifier(n_estimators=5)
        self.gb_reg.fit(self.X_train, self.y_train)
        self.train_performance(self.gb_reg)
        self.test_performance(self.gb_reg)

    def xgb_(self):
        self.xgb_reg = XGBClassifier()
        self.xgb_reg.fit(self.X_train, self.y_train)
        self.train_performance(self.xgb_reg)
        self.test_performance(self.xgb_reg)

    def svm_c(self):
        self.svm_reg = SVC(kernel='rbf', probability=True)
        self.svm_reg.fit(self.X_train, self.y_train)
        self.train_performance(self.svm_reg)
        self.test_performance(self.svm_reg)

    def training(self):
        print("=========KNN===========")
        self.knn()
        print("=========NB===========")
        self.nb()
        print("=========LR===========")
        self.lr()
        print("=========DT===========")
        self.dt()
        print("=========RF===========")
        self.rf()
        print("=========AdaBoost===========")
        self.ada()
        print("=========GB===========")
        self.gb()
        print("=========XGB===========")
        self.xgb_()
        print("=========SVM===========")
        self.svm_c()

    def predictions(self):
        self.models = {"KNN": self.knn_reg,
                       "NB": self.nb_reg,
                       "LR": self.lr_reg,
                       "DT": self.dt_reg,
                       "RF": self.rf_reg,
                       "AdaBoost": self.ada_reg,
                       "GB": self.gb_reg,
                       "XGB": self.xgb_reg,
                       "SVM": self.svm_reg
                       }

        self.proba_dict = {}
        for name, model in self.models.items():
            self.proba_dict[name] = model.predict_proba(self.X_test)

    def roc_auc_curve(self):
      plt.figure(figsize=(10, 7))
      auc_scores = {}   # store AUC values for each model
      for name, pred_proba in self.proba_dict.items():
        # Micro-average ROC curve
        fpr_micro, tpr_micro, _ = roc_curve(
            self.y_test_bin.ravel(),
            pred_proba.ravel()
        )
        auc_micro = auc(fpr_micro, tpr_micro)
        auc_scores[name] = auc_micro   
        print(f"Model: {name} | Micro AUC: {auc_micro:.4f}")
        plt.plot(fpr_micro, tpr_micro, label=f"{name} (AUC = {auc_micro:.4f})")
      best_model = max(auc_scores, key=auc_scores.get)
      best_auc = auc_scores[best_model]

      print("\n========== BEST MODEL ==========")
      print(f"Best Model: {best_model}  Best Micro AUC: {best_auc:.4f}")
      print("================================\n")

      plt.plot([0, 1], [0, 1], 'k--')
      plt.xlabel("False Positive Rate")
      plt.ylabel("True Positive Rate")
      plt.title("Multiclass ROC Curve for All Models")
      plt.legend()
      plt.show()
    
    def svm_gridsearch(self):
      print("\n=========== Running GridSearchCV for SVM ===========")

      param_grid = {
        "C": [0.1, 1, 10],
        "kernel": ["rbf", "linear", "poly","sigmoid"],
        "class_weight": [None, 'balanced']
      }

      svm = SVC(probability=True)
      grid = GridSearchCV(
        estimator=svm,
        param_grid=param_grid,
        scoring="accuracy",
        cv=10,
      )

      grid.fit(self.X_small, self.y_small)

      self.best_svm_params = grid.best_params_
      self.best_svm_model = grid.best_estimator_

      print("\n===== BEST PARAMETERS FOUND FOR SVM =====")
      print(self.best_svm_params)
      print("Best CV Accuracy:", grid.best_score_)

      print("\n===== Train Performance (Best SVM) =====")
      self.train_performance(self.best_svm_model)

      print("\n===== Test Performance (Best SVM) =====")
      self.test_performance(self.best_svm_model)

      # Add to model list for ROC curve
      self.models["SVM_Tuned"] = self.best_svm_model
      self.proba_dict["SVM_Tuned"] = self.best_svm_model.predict_proba(self.X_test)

      print("\nTuned SVM added as 'SVM_Tuned' for ROC curve.\n")


if __name__ == '__main__':
    dataset_path = "C:\\Users\\Rajesh\\Downloads\\ML Mini Project-1\\mnist_train.csv"

    obj = MNIST(dataset_path)
    obj.training()
    obj.predictions()
    obj.roc_auc_curve()

