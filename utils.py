import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np

def save_metrics(y_pred_class, y_test, file_path):
    metrics_dict = {}
    metrics_dict['f1-score'] = f1_score(y_pred_class, y_test)
    metrics_dict['precision'] = precision_score(y_pred_class, y_test)
    metrics_dict['recall'] = recall_score(y_pred_class, y_test)
    metrics_dict['accuracy'] = accuracy_score(y_pred_class, y_test)
    df_metrics = pd.DataFrame(metrics_dict, index=[0])
    df_metrics.to_csv(file_path + '/metrics.csv', index=False)
    
    report = classification_report(y_pred_class, y_test, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(file_path + '/classification_report.csv')
    
    cm = confusion_matrix(y_test, y_pred_class)
    df_cm = pd.DataFrame(cm, columns=np.unique(y_test), index = np.unique(y_test))
    df_cm.to_csv(file_path + '/confusion_matrix.csv')


def print_metrics(pred_tag, y_test):
    print("F1-score: ", f1_score(pred_tag, y_test))
    print("Precision: ", precision_score(pred_tag, y_test))
    print("Recall: ", recall_score(pred_tag, y_test))
    print("Acuracy: ", accuracy_score(pred_tag, y_test))
    print()
    # classification report
    print(classification_report(pred_tag, y_test))


def plot_confusion_matrix(y_test, y_pred_class, file_path):
    cm = metrics.confusion_matrix(y_test, y_pred_class)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    all_sample_title = 'Accuracy Score: {:.4f}'.format(metrics.accuracy_score(y_test, y_pred_class))
    plt.title(all_sample_title, size = 15)
    plt.tight_layout()
    plt.savefig(file_path, format='png', transparent=True)
    plt.show()


import matplotlib.pyplot as plt

def plot_and_save_loss(history, filepath):
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.title('Loss')
    plt.legend()
    plt.savefig(filepath, format='png', transparent=True)
    plt.show()

def plot_and_save_accuracy(history, filepath):
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='test')
    plt.title('Accuracy')
    plt.legend()
    plt.savefig(filepath, format='png', transparent=True)
    plt.show()
