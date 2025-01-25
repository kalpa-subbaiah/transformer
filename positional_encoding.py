import torch
import torch.nn as nn
import math
import input_embedding


class PosEnc(nn.Module):
    def __init__(self, seq_len,embed_dim):
        super(PosEnc, self).__init__()

        position_enc = torch.zeros(seq_len, embed_dim)
        pos = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0)/ embed_dim))

        position_enc[:, 0::2] = torch.sin(pos * div_term)
        position_enc[:, 1::2] = torch.cos(pos * div_term)

        self.register_buffer("pos_enc", position_enc.unsqueeze(0))

    def forward(self, inputs):
        return inputs + self.pos_enc[:, :inputs.size(1)]
    

if __name__ == "__main__":

    #Define parameters 
    vocab_size = 10000   # Vocabulary size
    embed_dim= 6         # Embedding Dimension
    seq_len = 4          # Sequence Length
    batch_size = 2       # Bacth size
  
    #Input Embedding

    sentences = ["I do love programming", "Transformers are very powerful"]
    print("INPUT TEXT : \n " , sentences)

    print("\n\nTOKENS : \n " , [sentence.split(" ") for sentence in sentences])

    #Example input tensor of shape (batch_size, seq_len) 
    x = torch.randint(0,vocab_size-1,(batch_size, seq_len))
    print("\n\nTOKENIZED INDEX  : \n ", x)  

    # Initialize the InputEmbedding class
    embedding_layer = input_embedding.InputEmbedding(vocab_size, embed_dim) 

    output_embeddings = embedding_layer(x)

    print("\n\nINPUT SHAPE : (BATCH SIZE, SEQ_LEN) \n ", x.shape),
    print("\n\nOUTPUT EMBEDDING SHAPE : (BATCH SIZE, SEQ_LEN, EMBEDDING_DIM)\n " , output_embeddings.shape)

    print("\n\nEMBEDDING VECTORS  : \n " , output_embeddings)

    #Position Encoding

    pos_enc = PosEnc(seq_len,embed_dim)

    position_encoding = pos_enc(output_embeddings)

    print("\n\n POSITIONAL ENCODING SHAPE: " , position_encoding.shape)
    print("\n\n POSITIONAL ENCODING output: " , position_encoding)


