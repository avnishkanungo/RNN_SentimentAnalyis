# RNN_SentimentAnalyis

## Table of Contents

- [Dataset Preparation](#dataset-preparation)
- [Model Architecture Used](#model-architecture)
- [References and Notes](#references-and-notes)



### Dataset Prepration
We have implemnented the sentiment analysis on two datasets:
1. data.csv provided in the repo
2. Sentiment Dataset provided in this [link](https://engineering.purdue.edu/kak/distDLS/text_datasets_for_DLStudio.tar.gz)
The data for both the datasets are stored in csv format, which is then prepared and ingested into the model using the pytorch dataset and dataloader classes. The detailed implemenation for both the datasets can be found in the documentation.
Along with this for the prepration of the text to convert the same into word embeddings to be passed into the dataloader we use the DistilBert model from the Transformers library provided by huggingface.

## Model Architecture Used
The models implementated can be found below, these are extesions of RNNs for handling problems faced with RNNs:
1. GRU: Gated Recurrent Unit (GRU) is a type of recurrent neural network (RNN) architecture designed to address the vanishing gradient problem often encountered in traditional RNNs. GRU models are characterized by their gating mechanisms, which regulate the flow of information within the network.
2. BiDirectional GRU: Bidirectional Gated Recurrent Unit (GRU) models are an extension of the standard GRU architecture that enables the network to process input sequences in both forward and backward directions. Introduced as an enhancement to traditional unidirectional RNNs, bidirectional GRUs leverage information from past and future time steps simultaneously, allowing the model to capture dependencies from both directions.In bidirectional GRU models, the input sequence is fed into two separate GRU layers: one processing the sequence in the forward direction and the other in the backward direction

The outputs from both these model implemenatations can be found in the attached PDF.

## References and Notes
- Implementation details, required libraries and performance evaluation of the models can be found in the attached PDF.






