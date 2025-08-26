# Flow Cytometry ML Pipeline — CS279 Project

**Evaluating the Impact of Dimensionality Reduction on Flow Cytometry ML Models**

A Stanford CS279 machine learning project that builds and compares dimensionality reduction methods to improve interpretability and performance of classifiers trained on high-dimensional flow cytometry data for white blood cell classification.

Link to research paper for final assignment in CS279 course: https://docs.google.com/document/d/1IzTpY50HBXuPKdhnfuKxsrRETyFqfIDhkzTg1cvAP3Y/edit?usp=sharing

## 📊 Project Overview

This project implements and evaluates various machine learning approaches for classifying white blood cell types from flow cytometry data. The pipeline includes:

- **Dimensionality Reduction Methods**: PCA, LDA, and autoencoders
- **Classification Algorithms**: AdaBoost, Gradient Boosting, KNN, Naive Bayes, Random Forest, SVC
- **Performance Comparison**: Control vs. autoencoder-based feature extraction
- **Cell Type Classification**: Neutrophil, Monocyte, T-cell, Eosinophil, B-cell

## 📁 Repository Structure

```
cs279-flow-cytometry-ml/
├── Train_code/          # Training scripts and utilities
├── Test_code/           # Testing and evaluation scripts
├── train_data/          # Training datasets
├── test_data/           # Test datasets organized by patient
├── output/              # Generated results and visualizations
└── README.md           # This project overview
```

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/dibdin/Flow-Cytometry-ML-CS279-project-.git
cd Flow-Cytometry-ML-CS279-project-
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Training Pipeline
```bash
cd Train_code
python train_main.py
```

### 4. Run Testing and Evaluation
```bash
cd Test_code
python test_labeled_unstained_balanced.py
```

## 🔬 Methodology

### Data Processing
- **Flow Cytometry Data**: High-dimensional cell measurements
- **Cell Types**: 5 white blood cell types (Neutrophil, Monocyte, T-cell, Eosinophil, B-cell)
- **Patients**: Multiple patient datasets (CRF022, CRF034, etc.)

### Dimensionality Reduction
- **Autoencoders**: Neural network-based feature extraction
- **PCA**: Principal Component Analysis
- **LDA**: Linear Discriminant Analysis

### Classification Models
- **AdaBoost**: Adaptive Boosting
- **Gradient Boosting**: Gradient Boosting Machines
- **KNN**: K-Nearest Neighbors
- **Naive Bayes**: Gaussian Naive Bayes
- **Random Forest**: Ensemble method
- **SVC**: Support Vector Classification

## 📈 Results & Highlights

### Key Findings
- **Enhanced Performance**: Autoencoder-based feature extraction improves classification accuracy
- **Computational Efficiency**: Reduced dimensionality enables faster training and inference
- **Interpretability**: Lower-dimensional representations maintain biological relevance
- **Robustness**: Consistent performance across different patient datasets

## 👨‍🔬 About the Author

**Diba Dindoust** — ML researcher at Stanford University

Focused on applying AI to health challenges, including:
- **Neonatal Outcomes Research**: Serum biomarker analysis for gestational age prediction
- **Non-hormonal Contraceptive Screening**: Machine learning for drug discovery
- **Health Accessibility**: AI solutions for low-resource settings
- **Flow Cytometry**: Automated cell classification and analysis

### Contact & Links
- **GitHub**: [dibdin](https://github.com/dibdin)
- **Email**: dibadindoust@stanford.edu

## 📚 References

This project was developed as part of Stanford CS279 (Machine Learning) coursework, focusing on practical applications of dimensionality reduction and classification in biomedical data analysis.

---

*Built with ❤️ for advancing healthcare through machine learning*
