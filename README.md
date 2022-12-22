# ML-project2-text-classification

This project aims at making supervised binary classification of tweets according to the emotion given by this phrase. These tweets initially contained positive smiley ”:)” or negative smiley ”:(”. These emojis has been removed from the tweets and the tweets has been labeled according to the emoji removed.

Please refer to our report, ML___Project_2.pdf, to understand our methodology and achievements.

## Create the predictions : `run.py`

The method used to create the submission file used on AIcrowd is 'fine-tuning pre-trained transformer models from HuggingFace'. In order to create the submission file posted on AIcrowd, you have to run the file `run.py`. It is a cleaned version of our notebook about the choosen method. 

Here are the steps to follow : 

* Clone the repository with the command : `git clone https://github.com/TheGreatJanus/ML-project2-text-classification` 
* Run the `run.py`script.

Please note that we saved our model weights in the file model.pt in order for you to recreate the submission file file `run.py` without redoing all the training that takes about 45 minutes to rune in Google Colab. If you want to run all the training, you can replace the `line 85` of `run.py` by `trainer.train()`. It takes about 45 minutes on Google Colab to do the all training. You can reduce the amount of data used for training in `line 79` (taking 5000 data instead of 20,000 will reduce the execution time to about 12 minutes. If you do not have a GPU on your laptop, we strongly recommend that you run this script on Google Colab to take advantage of their GPU.


## Other content of the repository

In this repository you can find all the notebooks summarising our exploratory work to get the best accuracy on our prediction. We decided for simplicity to create different notebooks as the preprocessing and text vectorization is different for the several techniques we tried.

### `vizualize-data.ipynb` 
Firstly, in this workbook, we have made an explanatory study of the data to better understand it. In particular, we have displayed the word clouds to visualise the difference between the words used in positive tweets and those used in negative tweets.

### `preprocessing-pipeline.ipynb`

In this notebook, we evaluated the most appropriate preprocessing methods to use in order to best prepare the data for model training and prediction. We also placed all the useful definitions on preprocessing in the python file `f_preprocessing.py` so that they can be used in other notebooks.

### `sklearn_predictions.ipynb`
In this notebook we evaluated the methods of : Logistic regression, Descision tree and a Simple Multi-layer Perceptron classifier. The ML library used is sklearn

### `bidirectional-LSTM_predictions.ipynb`
In this notebook we evaluated the method of a bidirectional-LSTM model : a neural network containing bidirectional LSTM layers. The ML libraries used are keras from tensorflow, as well as sklearn.

### `fine-tune-BERT_predictions.ipynb`
Finally, in this notebook, we evaluated a method of fine-tuning pre-trained transformer models from HuggingFace. The ML libraries used are transformers and datasets from HuggingFace, as well as sklearn.
