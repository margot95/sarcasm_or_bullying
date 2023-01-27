# sarcasm_or_bullying

bullying or sarcasm ?

We will try to make a big dataset out of two datasets downloaded from Kaggle. The first is: https://www.kaggle.com/datasets/danofer/sarcasm The second one is: https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification

We want later to train a LSTM neural network to classify a new comment/tweet into "sarcasm" or "cyberbullying". It can not be "neutral", let's imagine we are in a situation on a social media platform when someone has reported a comment as being harmful, and that our algorithm needs to say "yes, it is bullying" or "no, it is only dark humor/sarcasm". The idea comes from the fact that it is super difficult for people to sit and agree on where to draw the line between what is okay or not, so algorithms might come in handy!

In this notebook, I'm doing EDA and basically taking decisions on what information is useful to keep for feeding the model VS what information is noise, for a classification task.
