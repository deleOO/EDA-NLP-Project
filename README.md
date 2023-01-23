# Easy Data Augmentation (EDA) for Text Classification with Disaster Tweets
Experimental report for the course of Introduction to Natural Language Processing, University of Trento, 2021-22.

## General info
In this experimental study, we aim to evaluate the impact of the Easy Data Augmentation (EDA) framework (https://arxiv.org/abs/1901.11196) on the performance of various text classification models. The main goal of the study is to replicate the results of the original work by applying EDA techniques using the same settings and parameters as suggested for a small dataset by the authors. In particular, through this examination, we aim to determine the extent to which the performance improvements reported in the original work can be replicated on datasets that have not been used as benchmarks by the authors. Furthermore, our study seeks to establish the generalizability of the EDA framework to different datasets and its potential to enhance the performance of tweet classification models.

More details are in the report that you can find here:
https://drive.google.com/file/d/1lmunFlgFaavtm9HzTtn7w2K5oeaSEmZ-/view?usp=sharing 

## Dataset
The dataset used in this study is the Disaster Tweets from a Kaggle competition (https://www.kaggle.com/competitions/nlp-getting-started/overview). This dataset contains 7613 tweets that are labeled as either disaster-related or not. The dataset is available on Kaggle and it is also included in the repository.

## Models
Models used in this study are the following:
- Support Vector Machines (SVM)
- Multinomial Naive Bayes (MNB)
- Recurrent Neural Network (RNN) with LSTM cells based on the architecture proposed by Liu et al. (https://arxiv.org/abs/1605.05101)


## Setup
Tested on Windows 10 Home 20H2 and Google Colab.

### Local setup

1. Clone the repository
```
git clone https://github.com/deleOO/EDA-NLP-Project.git
```

2. Create a virtual environment
```
python -m venv .venv
.venv\Scripts\activate
```

3. Install the requirements
```
pip install -r requirements.txt
pip install textattack
```

4. Run download_GloVe.py to download the GloVe embeddings used by the RNN model
```
python download_GloVe.py
```

5. Use the jupyter notebooks in the repository to play with the code and the models. 
```
- `tweet-disaster-ML.ipynb`
- `tweet-disaster-LSTM.ipynb`
```
In the ML version you can train Naive Bayes and SVM models (but also Logistic Regression and Random Forest), while in the LSTM version you can train the RNN model. The notebooks are well commented and they contain all the necessary instructions to run the code.
### Google Colab setup
You can run the code on Google Colab. To do so, you have just to follow the instructions in the notebook that we provide.

1. ML version: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1jRHEGByWHrwNM3fWKfl3Nzx4La-zMAjP/view?usp=sharing)

2. LSTM version: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1iCqfcW3rqhIJt6sr1nFu14VW9ZTF3wYj/view?usp=sharing)




