import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix, roc_auc_score, make_scorer

#RandomForest
class ChurnModel:
    def __init__(self, file_path):
        """
        Inicializa la clase ChurnModel con la ruta al archivo de datos.
        """
        self.file_path = file_path
        self.df = self.load_data()
        self.X = self.df.drop(columns='attrition_flag')
        self.y = self.df['attrition_flag']
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data()
        self.classifier_rf = RandomForestClassifier(random_state=77300)
        self.classifier_gb = GradientBoostingClassifier()
        self.grid_search_rf = None

    def load_data(self):
        """
        Carga los datos desde un archivo Excel y los devuelve como un DataFrame.
        """
        return pd.read_excel(self.file_path)

    def split_data(self, test_size=0.20):
        """
        Divide los datos en conjuntos de entrenamiento y prueba.
        """
        np.random.seed(43)
        return train_test_split(self.X, self.y, test_size=test_size, stratify=self.y)

    def train_random_forest(self, n_trees=[50, 100, 150, 200, 250, 300]):
        """
        Entrena el modelo de Random Forest usando GridSearch para optimizar los hiperparámetros.
        """
        score_func = make_scorer(roc_auc_score, greater_is_better=True)
        param_grid = [{'n_estimators': n_trees}]

        grid_search = GridSearchCV(estimator=self.classifier_rf, param_grid=param_grid, cv=5,
                                   scoring=score_func, return_train_score=True)
        self.grid_search_rf = grid_search.fit(self.X_train, self.y_train)

    def predict(self):
        """
        Realiza predicciones sobre el conjunto de prueba.
        """
        y_pred_prob = self.grid_search_rf.predict_proba(self.X_test)[:, 1]
        return np.where(y_pred_prob > 0.5, 1, 0), y_pred_prob

    def evaluate_model(self):
        """
        Evalúa el modelo y muestra las métricas.
        """
        y_pred, y_pred_prob = self.predict()

        cm = confusion_matrix(self.y_test, y_pred)
        print("Confusion matrix:\n", cm)
        print("Accuracy:", self.custom_accuracy_score(cm))
        print("Sensitivity (Recall):", self.custom_sensitivity_score(cm))
        print("Specificity:", self.custom_specificity_score(cm))
        print("Positive Predictive Value (Precision):", self.custom_ppv_score(cm))
        print("Negative Predictive Value:", self.custom_npv_score(cm))

        self.plot_roc(y_pred_prob)

    def plot_roc(self, y_pred_prob):
        """
        Grafica la curva ROC.
        """
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_prob, pos_label=1)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (AUC = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([-0.001, 1.001])
        plt.ylim([-0.001, 1.001])
        plt.xlabel('1-Specificity (False Negative Rate)')
        plt.ylabel('Sensitivity (True Positive Rate)')
        plt.title('ROC curve')
        plt.legend(loc="lower right")
        plt.show()

    @staticmethod
    def custom_sensitivity_score(cm):
        """
        Calcula la sensibilidad (recall).
        """
        tn, fp, fn, tp = cm.ravel()
        return tp / (tp + fn)

    @staticmethod
    def custom_specificity_score(cm):
        """
        Calcula la especificidad.
        """
        tn, fp, fn, tp = cm.ravel()
        return tn / (tn + fp)

    @staticmethod
    def custom_ppv_score(cm):
        """
        Calcula el valor predictivo positivo (precision).
        """
        tn, fp, fn, tp = cm.ravel()
        return tp / (tp + fp)

    @staticmethod
    def custom_npv_score(cm):
        """
        Calcula el valor predictivo negativo.
        """
        tn, fp, fn, tp = cm.ravel()
        return tn / (tn + fn)

    @staticmethod
    def custom_accuracy_score(cm):
        """
        Calcula la precisión (accuracy).
        """
        tn, fp, fn, tp = cm.ravel()
        return (tn + tp) / (tn + tp + fn + fp)

