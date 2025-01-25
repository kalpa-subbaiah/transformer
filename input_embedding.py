import torch
import torch.nn as nn 
import numpy as np


class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        """
        Initializes the InputEmbedding layer.

        Parameters:
        vocab_size (int): Size of the vocabulary (i.e., the number of unique tokens)
        embed_dim (int): Dimensionality of the embedding space (i.e., size of the embedding vectors)
        """

        super(InputEmbedding, self).__init__()

        #Embedding layer for converting tokens to dense layers
        self.embedding=nn.Embedding(vocab_size, embed_dim)

    def forward(self, x):
        """
        Forward pass to get the embedding representation of input tokens.

        Parameters:
        x (tensor) :  A tensor of token indices with shape (batch_size, seq_len),
                      where each value represents a token's index in the vocabulary.

        Returns:
        Tensor: A tensor of shape (batch_size, seq_len, embed_dim) representing
                the input token's embeddings.
        """
        return self.embedding(x)
        

if __name__ == "__main__":

    #Define parameters 
    vocab_size = 10000   # Vocabulary size
    embed_dim= 6         # Embedding Dimension
    seq_len = 4          # Sequence Length
    batch_size = 2       # Batch Size

    # Initialize the InputEmbedding class
    embedding_layer = InputEmbedding(vocab_size, embed_dim)

    sentences = ["I do love programming", "Transformers are very powerful"]
    print("INPUT TEXT : \n " , sentences)

    print("\n\nTOKENS : \n " , [sentence.split(" ") for sentence in sentences])


    #Example input tensor of shape (batch_size, seq_len) 
    x = torch.randint(0,vocab_size-1,(batch_size, seq_len))
    print("\n\nTOKENIZED INDEX  : \n ", x)   

    output_embeddings = embedding_layer(x)

    print("\n\nINPUT SHAPE : (BATCH SIZE, SEQ_LEN) \n ", x.shape),
    print("\n\nOUTPUT EMBEDDING SHAPE : (BATCH SIZE, SEQ_LEN, EMBEDDING_DIM)\n " , output_embeddings.shape)

    print("\n\nEMBEDDING VECTORS  : \n " , output_embeddings)



















