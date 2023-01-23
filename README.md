# Easy Data Augmentation (EDA) for Text Classification with Disaster Tweets
Experimental report for the course of Introduction to Natural Language Processing, University of Trento, 2021-22.

## General info
In this experimental study, we aim to evaluate the impact of the Easy Data Augmentation (EDA) framework\cite{wei2019eda} on the performance of various text classification models. The main goal of the study is to replicate the results of the original work by applying EDA techniques using the same settings and parameters as suggested for a small dataset by the authors. In particular, through this examination, we aim to determine the extent to which the performance improvements reported in the original work can be replicated on datasets that have not been used as benchmarks by the authors. Furthermore, our study seeks to establish the generalizability of the EDA framework to different datasets and its potential to enhance the performance of tweet classification models.

## Dataset
The dataset used in this study is the Disaster Tweets from a Kaggle competition \cite{howard2019natural}. This dataset contains 7613 tweets that are labeled as either disaster-related or not. The dataset is available on Kaggle and it is also included in the repository.

## Models
Models used in this study are the following:
- Support Vector Machines (SVM)
- Multinomial Naive Bayes (MNB)
- Recurrent Neural Network (RNN) with LSTM cells based on the architecture proposed by Liu et al. \cite{liu2016recurrent}


## Setup
Tested on Windows 10 Home 20H2 and Google Colab.

### Local setup

1. Clone the repository
```
git clone xxx
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

5. Use the jupyter notebooks in the repository to play with the code and the models
```
- `Cifar100_SimCLR_Implementation.ipynb`
- `Cifar100_Downstream_task_LINEAR.ipynb`
- `Cifar100_Downstream_task_KNN.ipynb
```

### Google Colab setup
You can run the code on Google Colab. To do so, you have just to follow the instructions in the notebook that we provide.

1. Open the notebook `Cifar100_SimCLR_Implementation.ipynb` in Google Colab




