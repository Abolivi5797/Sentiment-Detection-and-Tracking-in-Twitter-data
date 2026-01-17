# Sentiment Detection and Tracking in Twitter Data Description:
This repository contains the code, experiments, and documentation for the **Sentiment Detection and Tracking in Social Media Streams** project. The project uses **Twitter data** to develop a resilient system capable of identifying whether sentiments expressed in social media text are **positive, negative, or neutral**.

## Project Overview

The objective of this project is to conduct a comprehensive comparative analysis of **traditional machine learning, deep learning, and state-of-the-art transformer models** on real-world Twitter data.

Given the informal, noisy, and dynamic nature of Twitter content, the project investigates how different modeling techniques perform when combined with **rigorous text preprocessing and feature engineering**. The work follows a progressive modeling approach, beginning with traditional machine learning algorithms, extending to deep learning architectures, and finally incorporating **transformer-based models**.

Experiments are conducted on benchmark Twitter datasets to evaluate model performance across multiple sentiment classes. Through this comparative analysis, the project highlights the strengths and limitations of each approach and demonstrates how advanced transformer models such as **RoBERTa** can significantly improve sentiment classification accuracy when paired with effective preprocessing strategies.

## Key Features

- **Data Preprocessing:**  
  Rigorous text cleaning including the removal of URLs, hashtags, mentions, and punctuation, along with tokenization, lowercase conversion, stop-word removal, and lemmatization to ensure data quality.

- **Feature Extraction:**  
  Implementation of **TF-IDF (Term Frequency–Inverse Document Frequency)** for traditional models and **pre-trained word embeddings** (e.g., GloVe, WordPiece) for neural network architectures.

- **Machine Learning Models:**  
  Implementation and comparative evaluation of traditional classifiers (**Naive Bayes, Random Forest, Support Vector Machine**) and deep learning models (**CNN, RNN, BiLSTM**), alongside transformer-based approaches.

- **State-of-the-Art Transformers:**  
  Fine-tuning of the **RoBERTa (Robustly Optimized BERT Approach)** model, leveraging its bidirectional contextual understanding to achieve superior sentiment analysis performance.

- **Hyperparameter Tuning:**  
  Systematic optimization of learning rates, batch sizes, activation functions (ReLU, Softmax), and dropout rates to mitigate overfitting and enhance classification accuracy.

- **Advanced Evaluation Metrics:**  
  Performance assessment using **Accuracy, Macro F1 Score, and Average Recall** to ensure reliable evaluation across all sentiment classes.
