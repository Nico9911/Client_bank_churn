import numpy as np
import pandas as pd
class DataProcessor:
    """
    Clase para manejar el procesamiento de datos de un archivo Excel.
    Incluye la carga, transformación, codificación y guardado de datos.
    """

    def __init__(self, file_path):
        """
        Método constructor que inicializa la clase con la ruta del archivo y
        crea un atributo DataFrame vacío que será cargado posteriormente.

        :param file_path: Ruta del archivo Excel.
        """
        self.file_path = file_path  # Almacena la ruta del archivo de datos
        self.df = None  # Inicializa el DataFrame vacío

    def load_data(self):
        """
        Carga los datos desde un archivo Excel y los almacena en el atributo df.
        """
        # Lee el archivo Excel en el DataFrame 'df'
        self.df = pd.read_excel(self.file_path)

    def transform_data(self):
        """
        Aplica varias transformaciones al DataFrame, incluyendo:
        - Mapeo de valores en 'attrition_flag' a valores numéricos.
        - Creación de nuevas columnas calculadas.
        - Conversión de algunas columnas a variables categóricas.
        - Eliminación de columnas irrelevantes.
        """
        # Mapea 'attrition_flag' a valores numéricos: 1 para 'Existing Customer' y 0 para 'Attrited Customer'
        self.df['attrition_flag'] = self.df['attrition_flag'].map({
            'Existing Customer': 1,
            'Attrited Customer': 0
        })

        # Calcula la razón entre contactos y meses inactivos, creando una nueva columna 'activity_contact_ratio'
        self.df['activity_contact_ratio'] = self.df['contacts_count_12_mon'] / (self.df['months_inactive_12_mon'] + 1)

        # Calcula la razón de uso del crédito, creando una nueva columna 'credit_usage_ratio'
        self.df['credit_usage_ratio'] = self.df['total_revolving_bal'] / self.df['credit_limit']

        # Calcula la cantidad de transacciones por mes de antigüedad, creando la columna 'trans_per_month'
        self.df['trans_per_month'] = self.df['total_trans_ct'] / self.df['months_on_book']

        # Elimina la columna 'clientnum', ya que es solo un identificador y no aporta valor predictivo
        self.df.drop(columns=['clientnum'], inplace=True)

        # Define las columnas que serán convertidas a variables categóricas
        categorical_columns = ['gender', 'dependent_count', 'education_level', 'marital_status', 'income_category',
                               'card_category']

        # Convierte cada columna categórica a tipo 'category' para optimizar memoria y procesamiento
        for col in categorical_columns:
            self.df[col] = self.df[col].astype('category')

    def one_hot_encode(self):
        """
        Aplica la técnica de codificación one-hot a las variables categóricas del DataFrame.
        Esto convierte las variables categóricas en columnas binarias para cada categoría.
        """
        # Selecciona las columnas que no son numéricas y las convierte a variables dummy (codificación one-hot)
        self.df = pd.get_dummies(self.df,
                                 columns=self.df.select_dtypes(exclude=['int64', 'float64', 'boolean']).columns,
                                 drop_first=True)

    def save_to_excel(self, output_path):
        """
        Guarda el DataFrame procesado en un archivo Excel.

        :param output_path: Ruta y nombre del archivo Excel de salida.
        """
        # Guarda el DataFrame como un archivo Excel en la ubicación especificada
        self.df.to_excel(output_path, index=False, engine='openpyxl')
