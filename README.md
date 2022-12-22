# ML-project2-text-classification

This project aims at making supervised binary classification of tweets according to the emotion given by this phrase. These tweets initially contained positive smiley ”:)” or negative smiley ”:(”. These emojis has been removed from the tweets and the tweets has been labeled according to the emoji removed.

Please refer to our report, ML___Project_2.pdf, to understand our methodology and achievements.

## Create the predictions : `run.py`

The method used to create the submission file used on AIcrowd is 'fine-tuning pre-trained transformer models from HuggingFace'. In order to create the submission file posted on AIcrowd, you have to run the file `run.py`. It is a cleaned version of our notebook about the choosen method. 

Here are the steps to follow : 

* Download the dataset on the following link : https://www.aicrowd.com/challenges/epfl-ml-text-classification. 
* In `line 20`, `line 21` and `line 89` of the `run.py`script, change the data path to your local path to the files train_neg.txt, train_pos.txt and test_data.txt repsectively. 
* Run the `run.py`script.

Please note that it takes about 45 minutes on Google Colab to run this script. You can also reduce the amount of data used for training in `line 79` (taking 5000 data instead of 20,000 will reduce the execution time to about 12 minutes. If you do not have a GPU on your laptop, we strongly recommend that you run this script on Google Colab to take advantage of their GPU.


## Other content of the repository

In this repository you can find all the notebooks summarising our exploratory work to get the best accuracy on our prediction. We decided for simplicity to create different notebooks as the preprocessing and text vectorization is different for the several techniques we tried.

### `preprocessing-pipeline.ipynb`

First, in this notebook, we have made an explanatory study of the data to better understand them. We also evaluated the most appropriate preprocessing methods to use in order to best prepare the data for model training and prediction. We also placed all the useful definitions on preprocessing in the python file `f_preprocessing.py` so that they can be used in other notebooks.

### `sklearn_predictions.ipynb`
In this notebook we evaluated the methods of : Logistic regression, Descision tree and a Simple Multi-layer Perceptron classifier.  

### `bi-LSTM_predictions.ipynb`
In this notebook we evaluated the method of a bi-LSTM model : a neural network containing bidirectional LSTM layers.

### `fine-tune-BERT_predictions.ipynb`
Finally, in this notebook, we evaluated a method of fine-tuning pre-trained transformer models from HuggingFace.
