# Spotify App Reviews Sentiment Analysis

This project involves analyzing user reviews of the Spotify App collected from the Google Play Store between 1/1/2022 and 7/9/2022. We aim to gain insights into user sentiments and ratings through various data exploration, preprocessing, and modeling techniques.

## Dataset Information

- **Data Source:** [Spotify App Reviews 2022 on Kaggle](https://www.kaggle.com/datasets/mfaaris/spotify-app-reviews-2022)
- **Columns:**
  - `Time_submitted`: Timestamp when the review was submitted
  - `Review`: Customer's review about the application
  - `Rating`: Given rating for the application
  - `Total_thumbsup`: Number of people who found the review helpful
  - `Reply`: Reply to customer review

## Project Workflow

### 1. Data Exploration

We start by exploring the dataset to understand its characteristics, distributions, and potential insights.

### 2. Data Preprocessing

#### 2.1 Text Cleaning

We clean the text of each review using the following steps:
- Convert text to lowercase
- Remove punctuation
- Remove URLs
- Remove newline characters
- Transform informal words to formal ones
- Remove stopwords
- Perform lemmatization

### 3. Feature Engineering

We define our features and labels based on the cleaned text data.

### 4. Tokenization

We tokenize the reviews to convert them into numerical format for modeling.

### 5. Sequence Padding

To ensure that all reviews have the same length, we apply padding to the tokenized sequences.

### 6. Word Embeddings

We use pretrained 50-dimensional Glove word embeddings to represent words in the reviews. An embedding matrix is created for this purpose.

### 7. Data Modeling and Optimization

#### 7.1 Base Model

We start with a basic fully connected (dense) layer model (model_0) for multi-class classification, using accuracy as the metric.

#### 7.2 Architecture Tweak

We manually experiment with different numbers of layers and nodes in each layer to improve model accuracy.

#### 7.3 LSTM/GRU Benchmark

We replace one dense layer with an LSTM/GRU layer to benchmark the model's performance.

#### 7.4 Hyperparameter Tuning

- We manually fine-tune hyperparameters, including learning rates and batch sizes.
- We explore different optimizers to enhance model performance.

#### 7.5 Keras Random Search Tuner

We apply Keras random search tuner for automated hyperparameter tuning to benchmark and improve model performance.

### 8. Benchmarking with Traditional Machine Learning Models

We compare the performance of deep learning models with traditional machine learning models to assess their effectiveness in sentiment analysis.




