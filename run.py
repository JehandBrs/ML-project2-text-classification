# This run.py script must be runned by a gpu
# It takes around 30 minutes to run 

# install useful libraries if necessary
!pip install transformers
!pip install datasets
!pip install evaluate


# import libraries

import numpy as np
import pandas as pd
import transformers 
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer 
from sklearn.model_selection import train_test_split
from datasets import Dataset
import evaluate

# load data
train_neg = [tweet[:-1] for tweet in open('train_neg.txt').readlines()]
train_pos = [tweet[:-1] for tweet in open('train_pos.txt').readlines()]

# put data into lists and assemble all of it
# then separate it with a train test split
X, y = train_neg + train_pos, [0]*100000 + [1]*100000
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# convert data into the good form in order to be readable by the tranformers library Trainer
dataset_train = Dataset.from_list([{'label' : y_train[i], 'text':X_train[i]} for i in range(len(y_train))])
dataset_test = Dataset.from_list([{'label' : y_test[i], 'text':X_test[i]} for i in range(len(y_test))])


# Here are all models from hugging face we tried to fine tune
# We had different results and the best model has been kept not commented
# all the results from these models are on the report

# The models : 
#MODEL = 'roberta-base'
#MODEL = 'distilbert-base-uncased-finetuned-sst-2-english'
#MODEL = 'bert-base-uncased'
#MODEL = 'ProsusAI/finbert'
MODEL = 'finiteautomata/bertweet-base-sentiment-analysis'

# We define the tokenizer and the pretrained model : 
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# The first step before is to tokenize our tweets data into a BERT form
# The function below do that
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# We tokenize our train and test datasets
tokenized_dataset_train = dataset_train.map(tokenize_function, batched=True)
tokenized_dataset_test = dataset_test.map(tokenize_function, batched=True)


# Let's define the metrics to evalute the performance of our classifier
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# Let's build this trainer

# The training parameters and where we save it
training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")


# The trainer :
# It's an object from the transformers library that allows us to fine-tune BERT based models 
# with our own data. The training is very simple and optimized by the tranformers library.
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset_train.shuffle(seed=6).select(range(20000)),
    eval_dataset=tokenized_dataset_test.shuffle(seed=6).select(range(500)),
    compute_metrics=compute_metrics,
)

# Now let's train our Trainer
trainer.train()


# load test data and tokenize it 
TEST = [tweet[:-1] for tweet in open('test_data.txt').readlines()]
XX, yy = TEST, [0]*10000 
DATASET = Dataset.from_list([{'label' : yy[i], 'text':XX[i]} for i in range(len(yy))])
TOK_DATASET = DATASET.map(tokenize_function, batched=True)

# We make predictions on this test data using our fine-tuned model
pred = trainer.predict(TOK_DATASET)
pred_label = (pred.predictions[:,1]>pred.predictions[:,0]).astype(int)
classification = 2*pred_label-1

# We convert these predictions to a csv file to submit it to AIcrowd
DF = pd.DataFrame.from_dict({'Id': range(1, 10001), 'Prediction': classification.tolist()})
DF.to_csv('submissions.csv', index = False)
