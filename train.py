#Warning control
import warnings
warnings.filterwarnings('ignore')
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from encoder import Encoder
from dataset import EncoderDataset

dataset = EncoderDataset('nq_sample.tsv')

def train(dataset, num_epochs=10):
    embed_size = 512
    output_embed_size = 128
    max_seq_len = 64
    batch_size = 32

    n_iters = len(dataset) // batch_size + 1
    
    # define the question/answer encoders
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    question_encoder = Encoder(tokenizer.vocab_size, embed_size, output_embed_size)
    answer_encoder = Encoder(tokenizer.vocab_size, embed_size, output_embed_size)

    # define the dataloader, optimizer and loss function    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)    
    optimizer = torch.optim.Adam(list(question_encoder.parameters()) + list(answer_encoder.parameters()), lr=1e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        running_loss = []
        for idx, data_batch in enumerate(dataloader):

            # Tokenize the question/answer pairs (each is a batc of 32 questions and 32 answers)
            question, answer = data_batch
            question_tok = tokenizer(question, padding=True, truncation=True, return_tensors='pt', max_length=max_seq_len)
            answer_tok = tokenizer(answer, padding=True, truncation=True, return_tensors='pt', max_length=max_seq_len)
            if idx == 0 and epoch == 0:
                print(question_tok['input_ids'].shape, answer_tok['input_ids'].shape)
            
            # Compute the embeddings: the output is of dim = 32 x 128
            question_embed = question_encoder(question_tok)
            answer_embed = answer_encoder(answer_tok)
            if idx == 0 and epoch == 0:
                print(question_embed.shape, answer_embed.shape)
    
            # Compute similarity scores: a 32x32 matrix
            # row[N] reflects similarity between question[N] and answers[0...31]
            similarity_scores = question_embed @ answer_embed.T
            if idx == 0 and epoch == 0:
                print(similarity_scores.shape)
    
            # we want to maximize the values in the diagonal
            target = torch.arange(question_embed.shape[0], dtype=torch.long)
            loss = loss_fn(similarity_scores, target)
            running_loss += [loss.item()]
            if idx == n_iters-1:
                print(f"Epoch {epoch}, loss = ", np.mean(running_loss))
    
            # this is where the magic happens
            optimizer.zero_grad()    # reset optimizer so gradients are all-zero
            loss.backward()
            optimizer.step()

    return question_encoder, answer_encoder

Q_encoder, A_encoder = train(dataset, num_epochs=30)