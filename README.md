# text-sentiment-classifiers
Classifies portuguese text comments regarding its polarity (negative, neutral or positive).


Usage:

    python text_classifiers.py -c sgd -n True

## Repository content:
* **text_classifiers.py** file:

It is the main file which get all data from dataset folder and train/test a classifier, reporting its accuracy at the end of its execution.

#### Parameters:

  > **-c** (str): specify a classifier (options are: "_svm_", "_sgd_" or _"mnb"_).

  > **-n** (bool): specify to discard or not the neutral comments (options are: _True_ or _False_).


* **datasets** folder:

  - dataset_sample.txt:

    Data extracted from web used to train/test the classifier. Only a small sample of the original dataset is avaliable due its commercial pourpose.
    Each row is an instance containig comment and its polarity (0 = negative, 1=neutral, 2=positive), which is separated by a "/" character.

  - stop_words.txt:
  
    Set of portuguese stopwords used during the text pre-processing step.




