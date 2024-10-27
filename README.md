# Embedding Models for Question Answering

This repository contains code and resources for creating embedding models for question-answering (QA) tasks. The model uses Siamese encoders to generate embeddings for both questions and answers, employing a contrastive loss mechanism with cross-entropy loss to optimize and align the embeddings for semantically similar pairs.

## Overview

In question-answering systems, creating effective embeddings for both questions and answers is essential to enhance the retrieval and matching performance. This project implements Siamese neural networks with contrastive learning, where each branch of the network independently encodes questions and answers into a shared embedding space. The contrastive loss ensures that similar question-answer pairs are closer in the embedding space, while dissimilar pairs are farther apart.

## Model Architecture

The project employs Siamese encoders, where each encoder is a neural transformer encoder network that processes either the question or the answer. The two encoders share weights and are optimized together using contrastive loss by implementing cross entropy loss.

### Key Components
- **Siamese Encoders**: Identical encoders for questions and answers, sharing weights to produce comparable embeddings.
- **Contrastive Loss with Cross-Entropy**: The model minimizes the cross-entropy loss to pull similar question-answer embeddings closer and push dissimilar ones apart.
- **Embedding Space**: The final output is a vector representation for both questions and answers, which can be used in downstream QA tasks.

