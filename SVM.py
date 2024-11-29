import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer


class ChurnModel:
    def __init__(self, file_path):
        """
        Inicializa la clase ChurnModel con la ruta al archivo de datos.

        :param file_path: Ruta del archivo de Excel que contiene los datos de churn.
        """
        self.file_path = file_path
        self.df = self.load_data()
        self.y = self.df['attrition_flag']
        self.X = self.df.drop(columns='attrition_flag')

        # Establece la semilla para la reproducibilidad de los resultados
        np.random.seed(43)

        # Divide los datos en conjuntos de entrenamiento y prueba
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data()

        # Inicializa el clasificador SVM
        self.classifier_svm = Pipeline([
            ('scaler', StandardScaler()),  # Normaliza los datos
            ('SVM', SVC(probability=True))  # Clasificador SVM con probabilidad
        ])

    def load_data(self):
        """
        Carga los datos desde un archivo Excel y los devuelve como un DataFrame.

        :return: DataFrame con los datos cargados.
        """
        return pd.read_excel(self.file_path)

    def split_data(self, test_size=0.20):
        """
        Divide los datos en conjuntos de entrenamiento y prueba.

        :param test_size: Proporción de los datos que se utilizará para la prueba.
        :return: Tupla con los conjuntos de datos de entrenamiento y prueba.
        """
        return train_test_split(self.X, self.y, test_size=test_size, stratify=self.y)

    def train(self):
        """
        Entrena el modelo SVM utilizando los datos de entrenamiento.
        """
        self.classifier_svm.fit(self.X_train, self.y_train)

    def predict(self, class_threshold=0.6073):
        """
        Realiza predicciones sobre el conjunto de prueba utilizando el clasificador SVM.

        :param class_threshold: Umbral de decisión para clasificar las predicciones.
        :return: Predicciones y probabilidades de predicción.
        """
        # Predicciones de probabilidades
        y_pred_prob = self.classifier_svm.predict_proba(self.X_test)[:, 1]
        # Clasificación basada en el umbral
        y_pred = np.where(y_pred_prob > class_threshold, 1, 0)
        return y_pred, y_pred_prob

    def evaluate(self):
        """
        Evalúa el modelo y muestra las métricas.
        """
        y_pred, y_pred_prob = self.predict()

        # Matriz de confusión
        cm = confusion_matrix(self.y_test, y_pred)
        print("Confusion matrix:\n", cm)

        # Cálculo y visualización de métricas
        print("Accuracy:", self.custom_accuracy_score(cm))
        print("Sensitivity (Recall):", self.custom_sensitivity_score(cm))
        print("Specificity:", self.custom_specificity_score(cm))
        print("Positive Predictive Value (Precision):", self.custom_ppv_score(cm))
        print("Negative Predictive Value:", self.custom_npv_score(cm))

        self.plot_roc(y_pred_prob)

        # AUC
        print("AUC:", roc_auc_score(self.y_test, y_pred_prob))

    def plot_roc(self, y_pred_prob):
        """
        Grafica la curva ROC.

        :param y_pred_prob: Probabilidades de predicción.
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

        :param cm: Matriz de confusión.
        :return: Sensibilidad calculada.
        """
        tn, fp, fn, tp = cm.ravel()
        return tp / (tp + fn)

    @staticmethod
    def custom_specificity_score(cm):
        """
        Calcula la especificidad.

        :param cm: Matriz de confusión.
        :return: Especificidad calculada.
        """
        tn, fp, fn, tp = cm.ravel()
        return tn / (tn + fp)

    @staticmethod
    def custom_ppv_score(cm):
        """
        Calcula el valor predictivo positivo (precision).

        :param cm: Matriz de confusión.
        :return: Valor predictivo positivo calculado.
        """
        tn, fp, fn, tp = cm.ravel()
        return tp / (tp + fp)

    @staticmethod
    def custom_npv_score(cm):
        """
        Calcula el valor predictivo negativo.

        :param cm: Matriz de confusión.
        :return: Valor predictivo negativo calculado.
        """
        tn, fp, fn, tp = cm.ravel()
        return tn / (tn + fn)

    @staticmethod
    def custom_accuracy_score(cm):
        """
        Calcula la precisión (accuracy).

        :param cm: Matriz de confusión.
        :return: Precisión calculada.
        """
        tn, fp, fn, tp = cm.ravel()
        return (tn + tp) / (tn + tp + fn + fp)

    def hyperparameter_tuning(self):
        """
        Realiza la búsqueda de hiperparámetros utilizando GridSearchCV.
        """
        # Definir la métrica a optimizar
        score_func = make_scorer(roc_auc_score, greater_is_better=True)

        # Definir la cuadrícula de parámetros a probar
        param_grid = {
            'SVM__C': [0.8, 1, 1.5],
            'SVM__kernel': ['linear', 'rbf', 'poly']
        }

        # Crear el objeto GridSearchCV
        grid = GridSearchCV(self.classifier_svm, param_grid=param_grid, cv=5, scoring=score_func,
                            refit=True, verbose=3, return_train_score=True)

        # Ajustar el modelo
        grid_search = grid.fit(self.X_train, self.y_train)

        # Predicciones usando el mejor modelo encontrado
        grid_predictions = grid_search.predict(self.X_test)

        # Reportar resultados
        print(classification_report(self.y_test, grid_predictions))

