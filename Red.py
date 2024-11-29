import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scikeras.wrappers import KerasClassifier

class DataProcessor:
    def __init__(self, path):
        #Carga el archivo de excel a un Datframe de pandas
        self.df = pd.read_excel(path)
        #define las variables independientes (X) eliminando la columna 'attrition_flag'
        self.X = self.df.drop(columns='attrition_flag')
        #Define la variable dependiente (y) como la columna 'attrition_flag'
        self.y = self.df['attrition_flag']

    def split_data(self, test_size=0.2, random_state=43):
        #configura la semilla aleatoriamente para reproducibilidad
        np.random.seed(random_state)
        #divide los datos en conjuntos de entrenamiento y prueba, manteniendo la proporcionalidad de la variable y
        return train_test_split(self.X, self.y, test_size=test_size, stratify=self.y)

#Clase para evaluar las metricas
class Metrics:
    @staticmethod
    def plot_roc(y_test, y_pred_prob): #Calcula la curva ROC y el area bajo la curva (AUC)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob, pos_label=1, drop_intermediate=False)
        roc_auc = auc(fpr, tpr)
        #Dibuja la curva ROC
        plt.figure()
        lw = 2 #Ancho de linea
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
    def confusion_matrix_metrics(y_test, y_pred): #Calcula la matriz de confusion
        cm = confusion_matrix(y_test, y_pred)
        #Desglosa la matriz en valores TN,FP,FN,TP
        tn, fp, fn, tp = cm.ravel()
        #Retorna un diccionario con las metricas calculadas
        return {
            "Accuracy": (tn + tp) / (tn + fp + fn + tp),
            "Sensitivity (Recall)": tp / (tp + fn),
            "Specificity": tn / (tn + fp),
            "Positive Predictive Value (Precision)": tp / (tp + fp),
            "Negative Predictive Value": tn / (tn + fn)
        }
#Clase para crear y entrenar un modelo de red neuronal
class NeuralNetworkModel:
    def __init__(self, input_dim):
        #Crea un modelo secuencial de Keras con una capa oculta densa y una de salida
        self.model = keras.Sequential([
            layers.Dense(25, activation="relu", input_dim=input_dim, name="hidden-dense-25-layer"),
            layers.Dropout(0.3),#Capa de Dropout para reducir el sobreajuste
            layers.Dense(1, activation="sigmoid", name="output-layer")#Capa de salida con activacion sigmoide
        ])
        #Compila el modelo con la funcion de perdida binaria y el optimizador Adam
        self.model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])

    def get_pipeline(self):
        #Retorna un pipeline que estandariza los datos y entrena el modelo de red nauronal
        return Pipeline([
            ('standardize', StandardScaler()),#Escalador para normalizar los datos
            ('mlp', KerasClassifier(model=self.model, epochs=15, batch_size=64, validation_split=0.2))
        ])
#Clase para manejar el flujo completo de entrenamiento y evaluacion
class ChurnModel:
    def __init__(self, data_path):
        #Inicializa la clase de procesamiento de datos con la ruta al archivo de datos
        self.data_processor = DataProcessor(data_path)

    def run(self):
        # Divide los datos en conjutnos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = self.data_processor.split_data()

        # Crea la red neuronal y obtiene el pipeline
        model = NeuralNetworkModel(X_train.shape[1])
        pipeline = model.get_pipeline()

        # Entrena el modelo con los datos de entrenamiento
        pipeline.fit(X_train, y_train)

        # Predicciones de probabilidades sobre el conjunto de prueba
        y_pred_prob = pipeline.predict_proba(X_test)[:, 1]
        #Genera predicciones binarias usando umbral de 0.6
        y_pred = np.where(y_pred_prob > 0.6, 1, 0)

        # Genera la curva ROC
        Metrics.plot_roc(y_test, y_pred_prob)
        metrics = Metrics.confusion_matrix_metrics(y_test, y_pred)

        # Imprime las metricas
        print("Metrics of the Neural Network:\n")
        print(f"Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")

