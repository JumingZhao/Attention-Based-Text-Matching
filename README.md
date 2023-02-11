#Attention-Based Text Matching

[TOC]

## introduction

A text matching model. Can be trained on [SNIL dataset](https://nlp.stanford.edu/projects/snli/).  The model work as follow:

**·** Obtained input encoding by Bi-LSTM.

**·** Performed local inference by cross attention. Obtain the enhanced information of the two sentences.

**·** Let the enhanced information pass through a linear layer, Bi-LSTM, MLP in turn to predict
the relationship of the two sentences.



## Usage

download glove.6B.50d from [glove](https://nlp.stanford.edu/projects/glove/), put it in the root file. Download training data and test data(eg. SNIL dataset ). Then

```python
python main.py
```

