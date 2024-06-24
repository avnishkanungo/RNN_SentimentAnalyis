# -*- coding: utf-8 -*-
"""HW_9_New.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Tvo8YU08OWQ3WqDfa2MtrHkfecUfiJIo
"""

import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
from transformers import DistilBertTokenizer, DistilBertModel
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as tvt
import torch.optim as optim
import numpy as np
import numbers
import re
import math
import random
import copy
import matplotlib.pyplot as plt
import gzip
import pickle
import seaborn as sns
# import pymsgbox
import time
import logging

class SentencesDataset(Dataset):
    def __init__(self, file_path, test_train):
      self.df = pd.read_csv(file_path)

      # Tokenize and create word embedding for sentences
      sentences = [i for i in self.df['Sentence']]
      if test_train == 'train':
        sentences = sentences[:(4675)]
      elif test_train == 'test':
        sentences = sentences[4676:5843]
      else:
        raise ValueError("Invalid test_train value. Must be 'train' or 'test'.")
      word_tokenized_sentences = [sentence.split() for sentence in sentences]
      max_len = max([len(sentence) for sentence in word_tokenized_sentences])
      padded_sentences = [sentence + ['[PAD]'] * (max_len - len(sentence)) for sentence in word_tokenized_sentences]
      self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

      model_ckpt = "distilbert-base-uncased"
      distilbert_tokenizer = DistilBertTokenizer.from_pretrained(model_ckpt)
      bert_tokenized_sentences_ids = [ distilbert_tokenizer.encode(sentence , padding ='max_length',truncation =True ,max_length = max_len )for sentence in sentences ]

      vocab = {}
      vocab ['[PAD]'] = 0

      for sentence in padded_sentences:
        for token in sentence:
          if token not in vocab:
            vocab[token] = len(vocab)

      padded_sentences_ids = [[vocab[token] for token in sentence] for sentence in padded_sentences]

      distilbert_model = DistilBertModel.from_pretrained(model_ckpt)

      word_embeddings = []
      count_1 = 0
      for tokens in padded_sentences_ids :
        input_ids = torch.tensor(tokens).unsqueeze(0)
        with torch . no_grad ():
          outputs = distilbert_model(input_ids)
          count_1 += 1
          print(count_1)
        word_embeddings.append(outputs.last_hidden_state)

      subword_embeddings = []
      count_2 = 0
      for tokens in bert_tokenized_sentences_ids :
        input_ids = torch.tensor(tokens).unsqueeze(0)
        with torch . no_grad ():
          outputs = distilbert_model(input_ids)
          count_2 += 1
          print(count_2)
        subword_embeddings.append(outputs.last_hidden_state)

      self.embedding1 = word_embeddings
      self.embedding2 = subword_embeddings

      # Map sentiment labels to one-hot vectors
      self.sentiment_map = {'positive': [1, 0, 0],
                              'negative': [0, 1, 0],
                              'neutral': [0, 0, 1]}

    def __len__(self):
      return len(self.df)

    def __getitem__(self, idx):
      word_embedding = self.embedding1[idx]
      subword_embedding = self.embedding2[idx]
      sentiment = self.df.iloc[idx]['Sentiment']

      # Convert sentiment label to one-hot vector
      sentiment_label = self.sentiment_map[sentiment]
      sentiment_tensor = torch.tensor(sentiment_label)

      return word_embedding, subword_embedding, sentiment_tensor

train_dataset = SentencesDataset('/content/drive/MyDrive/HW9/data.csv', 'train')
torch.save(train_dataset, '/content/drive/MyDrive/HW9/dataset/train_SentencesDataset.pt')

test_dataset = SentencesDataset('/content/drive/MyDrive/HW9/data.csv', 'test')
torch.save(test_dataset, '/content/drive/MyDrive/HW9/dataset/testSentencesDataset.pt')

train_dataset = torch.load('/content/drive/MyDrive/HW9/dataset/train_SentencesDataset.pt')
test_dataset = torch.load('/content/drive/MyDrive/HW9/dataset/testSentencesDataset.pt')

len(train_dataset)

len(test_dataset)

# train_random_sampler = RandomSampler(train_dataset, num_samples=int(0.8 * len(train_dataset) ))
# test_random_sampler = RandomSampler(test_dataset, num_samples=int(0.2 * len(test_dataset) ))
# train_dataloader = DataLoader(train_dataset, batch_size=1, sampler=train_random_sampler)
# test_dataloader = DataLoader(test_dataset, batch_size=1, sampler=test_random_sampler)

x, y, z = next(iter(train_dataloader))

x.shape

dataset = torch.load('/content/drive/MyDrive/HW9/SentencesDataset.pt')

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

class GRUnet(nn.Module):
            def __init__(self, input_size, hidden_size, output_size, num_layers, drop_prob=0.2):
                super(GRUnet, self).__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                self.gru = nn.GRU(input_size, hidden_size, num_layers)
                self.fc = nn.Linear(hidden_size, output_size)
                self.relu = nn.ReLU()
                self.logsoftmax = nn.LogSoftmax(dim=1)

            def forward(self, x, h):
                out, h = self.gru(x, h)
                out = self.fc(self.relu(out[:,-1]))
                out = self.logsoftmax(out)
                return out, h

            def init_hidden(self):
                weight = next(self.parameters()).data
                #                                     batch_size
                hidden = weight.new(  self.num_layers,     1,         self.hidden_size   ).zero_()
                return hidden

class BiGRUnet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, drop_prob=0.2):
        super(BiGRUnet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = True  # Set bidirectional to True for bidirectional GRU
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # Multiply by 2 for bidirectional GRU
        self.relu = nn.ReLU()
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x, h):
        out, h = self.gru(x, h)
        # Concatenate the hidden states from both directions
        out = torch.cat((out[:, -1, :self.hidden_size], out[:, 0, self.hidden_size:]), dim=1)
        out = self.fc(self.relu(out))
        out = self.logsoftmax(out)
        return out, h

    def init_hidden(self):
        weight = next(self.parameters()).data
        num_directions = 2 if self.bidirectional else 1
        # Adjust shape for bidirectional GRU
        hidden = weight.new(self.num_layers * num_directions, 1, self.hidden_size).zero_()
        return hidden

def run_code_for_training_for_text_classification_with_GRU(net, display_train_loss=False):
            filename_for_out = "performance_numbers_" + str(1) + ".txt"
            FILE = open('/content/drive/MyDrive/HW9/saved_model'+filename_for_out, 'w')
            net.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
            ##  Note that the GRUnet now produces the LogSoftmax output:
            criterion = nn.NLLLoss()
            accum_times = []
            # optimizer = optim.SGD(net.parameters(),
            #              lr=1e-3, momentum=0.9)
            optimizer = optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999))
            start_time = time.perf_counter()
            training_loss_tally = []
            for epoch in range(1):
                print("")
                running_loss = 0.0
                for i, data in enumerate(train_dataloader):
                    review_tensor, bemb, sentiment = data
                    review_tensor = review_tensor[0]
                    sentiment = sentiment[0]
                    review_tensor = review_tensor.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
                    sentiment = sentiment.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
                    ## The following type conversion needed for MSELoss:
                    ##sentiment = sentiment.float()
                    optimizer.zero_grad()
                    hidden = net.init_hidden().to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
                    for k in range(review_tensor.shape[1]):
                        output, hidden = net(torch.unsqueeze(torch.unsqueeze(review_tensor[0,k],0),0), hidden)
                    # output, hidden = net(review_tensor, hidden)
                    ## If using NLLLoss, CrossEntropyLoss
                    # print(f'Output:{output}')
                    # print(f'Sentiment Argmax:{torch.argmax(sentiment.unsqueeze(0),1)}')
                    loss = criterion(output, torch.argmax(sentiment.unsqueeze(0),1))
                    ## If using MSELoss:
                    ## loss = criterion(output, sentiment)
                    running_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                    if i % 10 == 9:
                        avg_loss1 = running_loss / float(10)
                    if i % 200 == 199:
                        avg_loss = running_loss / float(200)
                        training_loss_tally.append(avg_loss)
                        current_time = time.perf_counter()
                        time_elapsed = current_time-start_time
                        print("[epoch:%d  iter:%4d  elapsed_time:%4d secs]     loss: %.5f" % (epoch+1,i+1, time_elapsed,avg_loss))
                        accum_times.append(current_time-start_time)
                        FILE.write("%.3f\n" % avg_loss)
                        FILE.flush()
                        running_loss = 0.0
            print("Total Training Time: {}".format(str(sum(accum_times))))
            print("\nFinished Training\n")
            return net.state_dict(), training_loss_tally, running_loss
            # torch.save(net.state_dict(), '/content/drive/MyDrive/HW9/saved_model/saved_model_gru')
            # if display_train_loss:
            plt.figure(figsize=(10,5))
            plt.title("Training Loss vs. Iterations")
            plt.plot(training_loss_tally)
            plt.xlabel("iterations")
            plt.ylabel("training loss")
#           plt.legend()
            plt.legend(["Plot of loss versus iterations"], fontsize="x-large")
            plt.savefig("training_loss.png")
            plt.show()

model_gru = GRUnet(768, hidden_size=512, output_size=3, num_layers=2)
trained_net_gru, avg_loss_gru, running_loss_gru = run_code_for_training_for_text_classification_with_GRU(model_gru, display_train_loss=True)

plt.figure(figsize=(10,5))
plt.title("Training Loss vs. Iterations")
plt.plot(avg_loss_gru)
plt.xlabel("iterations")
plt.ylabel("training loss")
#plt.legend()
plt.legend(["Plot of loss versus iterations"], fontsize="x-large")
# plt.savefig("training_loss2.png")
plt.show()

torch.save(trained_net_gru, '/content/drive/MyDrive/HW9/dataset/saved_model_gru')

model_bigru = BiGRUnet(768, hidden_size=512, output_size=3, num_layers=2)
trained_net_bigru, avg_loss_bigru ,running_loss_bigru = run_code_for_training_for_text_classification_with_GRU(model_bigru, display_train_loss=True) ### Issue is with the value of sentiment check out in the morning

torch.save(trained_net_bigru, '/content/drive/MyDrive/HW9/dataset/saved_model_bigru')

plt.figure(figsize=(10,5))
plt.title("Training Loss vs. Iterations")
plt.plot(avg_loss_bigru)
plt.xlabel("iterations")
plt.ylabel("training loss")
#plt.legend()
plt.legend(["Plot of loss versus iterations"], fontsize="x-large")
plt.savefig("training_loss.png")
plt.show()

def run_code_for_testing_text_classification_with_GRU(net, path):
            net.load_state_dict(torch.load(path))
            net.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
            classification_accuracy = 0.0
            negative_total = 0
            positive_total = 0
            confusion_matrix = torch.zeros(3,3)
            with torch.no_grad():
                for i, data in enumerate(test_dataloader):
                    wemb,review_tensor,sentiment = data
                    review_tensor = review_tensor[0]
                    sentiment = sentiment[0]
                    review_tensor = review_tensor.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
                    sentiment = sentiment.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
                    hidden = net.init_hidden().to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
                    for k in range(review_tensor.shape[1]):
                        output, hidden = net(torch.unsqueeze(torch.unsqueeze(review_tensor[0,k],0),0), hidden)
                    predicted_idx = torch.argmax(output).item()
                    gt_idx = torch.argmax(sentiment).item()
                    if i % 100 == 99:
                        print("   [i=%d]    predicted_label=%d  predicted     gt_label=%d\n\n" % (i+1, predicted_idx,gt_idx))
                    if predicted_idx == gt_idx:
                        classification_accuracy += 1
                    if gt_idx == 0:
                        negative_total += 1
                    elif gt_idx == 1:
                        positive_total += 1
                    confusion_matrix[gt_idx,predicted_idx] += 1

            print("\nOverall classification accuracy: %0.2f%%" %  (float(classification_accuracy) * 100 /float(i)))
            out_percent = np.zeros((3,3), dtype='float')
            out_percent[0,0] = "%.3f" % (100 * confusion_matrix[0,0] / float(negative_total))
            out_percent[0,1] = "%.3f" % (100 * confusion_matrix[0,1] / float(negative_total))
            out_percent[0,2] = "%.3f" % (100 * confusion_matrix[0,2] / float(negative_total))
            out_percent[1,0] = "%.3f" % (100 * confusion_matrix[1,0] / float(positive_total))
            out_percent[1,1] = "%.3f" % (100 * confusion_matrix[1,1] / float(positive_total))
            out_percent[1,2] = "%.3f" % (100 * confusion_matrix[1,2] / float(positive_total))
            out_percent[2,0] = "%.3f" % (100 * confusion_matrix[2,0] / float(negative_total))
            out_percent[2,1] = "%.3f" % (100 * confusion_matrix[2,1] / float(negative_total))
            out_percent[2,2] = "%.3f" % (100 * confusion_matrix[2,2] / float(negative_total))
            print("\n\nNumber of positive reviews tested: %d" % positive_total)
            print("\n\nNumber of negative reviews tested: %d" % negative_total)
            print("\n\nDisplaying the confusion matrix:\n")
            out_str = "                      "
            out_str +=  "%18s    %18s" % ('predicted negative', 'predicted positive')
            print(out_str + "\n")
            for i,label in enumerate(['true negative', 'true positive']):
                out_str = "%12s:  " % label
                for j in range(2):
                    out_str +=  "%18s" % out_percent[i,j]
                print(out_str)
            return confusion_matrix

output_gru = run_code_for_testing_text_classification_with_GRU(model_gru, '/content/drive/MyDrive/HW9/dataset/saved_model_gru')

sns.heatmap(output_gru, annot=True, cmap='Greens', fmt='g', xticklabels=['positive','negative', 'neutral'], yticklabels=['positive','negative', 'neutral'])
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
plt.title('Heatmap')
plt.show()

output_bigru = run_code_for_testing_text_classification_with_GRU(model_bigru, '/content/drive/MyDrive/HW9/dataset/saved_model_bigru')

sns.heatmap(output_bigru, annot=True, cmap='Greens', fmt='g', xticklabels=['positive','negative', 'neutral'], yticklabels=['positive','negative', 'neutral'])
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
plt.title('Heatmap')
plt.show()
