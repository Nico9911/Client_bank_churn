Client Bank Churn Prediction
This project aims to predict client churn for a financial institution that recently experienced a 15% reduction in active clients. Using machine learning techniques, the goal is to identify customers at risk of leaving and provide actionable insights for retention strategies.

Table of Contents
Project Overview
Dataset
Methodology
Models Used
Results
Technologies
How to Run
Future Work
Project Overview
Objective: Build a predictive model to classify clients likely to churn.
Context: The bank experienced a 15% drop in active clients during the last quarter.
Deliverables: Provide actionable insights for retention and a scalable machine learning solution.
Dataset
Size: 10,127 records, 21 features.
Variables:
Target: attrition_flag (Existing Customer or Attrited Customer).
Features: Demographic, financial, and activity-based variables, including:
customer_age, credit_limit, total_trans_amt, months_on_book, etc.
Source: Proprietary financial institution data.
Methodology
The project follows the CRISP-DM framework:

Business Understanding: Define objectives and success criteria.
Data Understanding: Explore and preprocess the dataset.
Modeling: Train multiple models (e.g., Neural Networks, SVM, Random Forest).
Evaluation: Compare metrics like accuracy, sensitivity, specificity, precision, and AUC.
Deployment: Provide model recommendations and implementation strategies.
Models Used
Artificial Neural Network (ANN):
Hidden layers: 1 with 25 neurons (ReLU activation).
Output layer: Sigmoid activation.
Optimizer: Adam.
Support Vector Machine (SVM):
Kernel: RBF.
Random Forest:
Trees: 100.
Results
ANN:
Accuracy: 92.20%
AUC: 0.94
SVM:
Accuracy: 92.49%
AUC: 0.95
Insights:
High sensitivity and precision demonstrate the models' effectiveness in identifying at-risk clients.
Technologies
Python
Libraries:
TensorFlow/Keras
Scikit-learn
Matplotlib, Seaborn (visualizations)
Google Colab
How to Run
Clone the repository:
bash
Copiar código
git clone https://github.com/Nico9911/Client_bank_churn.git
cd Client_bank_churn
Install dependencies:
bash
Copiar código
pip install -r requirements.txt
Run the main script:
bash
Copiar código
python main.py
(Optional) Open the Jupyter notebook for detailed analysis:
bash
Copiar código
jupyter notebook analysis.ipynb
Future Work
Hyperparameter optimization using GridSearch for models.
Explore additional features to improve accuracy.
Deploy the best model via a REST API.
Contributors
Nico9911: GitHub Profile
