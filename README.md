# sarcasm_or_bullying

bullying or sarcasm ?


data_preparation... notebook:
We will try to make a big dataset out of two datasets downloaded from Kaggle. The first is: https://www.kaggle.com/datasets/danofer/sarcasm The second one is: https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification

We want later to train a LSTM neural network to classify a new comment/tweet into "sarcasm" or "cyberbullying". It can not be "neutral", let's imagine we are in a situation on a social media platform when someone has reported a comment as being harmful, and that our algorithm needs to say "yes, it is bullying" or "no, it is only dark humor/sarcasm". The idea comes from the fact that it is super difficult for people to sit and agree on where to draw the line between what is okay or not, so algorithms might come in handy!

In this notebook, I'm doing EDA and basically taking decisions on what information is useful to keep for feeding the model VS what information is noise, for a classification task.

naives_bayes notebook:
In this notebook, I'm training a Multinomial Naive Bayes model on the dataset, performing cross-validation, visualizing our predictions (true positives, false positives, true negatives and false negatives) with a confusion matrix.
I also test the naive bayes model on two of my own sentences, does it classify them how I expect it to?
