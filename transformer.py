import math
import torch
import torch.nn as nn
import torch.optim as optim

# Positional Encoding
class PosEnc(nn.Module):
    def __init__(self, embed_dim, max_len):
        super(PosEnc, self).__init__()
        
        position_enc = torch.zeros(max_len, embed_dim)  # Create a zero tensor for positional encoding, shape: (max_len, embed_dim)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # Positions tensor from 0 to max_len-1, shape: (max_len, 1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))  # Compute the division term for sin and cos functions, shape: (embed_dim/2)

        position_enc[:, 0::2] = torch.sin(pos * div_term)  # Apply sine to even indices, shape: (max_len, embed_dim/2)
        position_enc[:, 1::2] = torch.cos(pos * div_term)  # Apply cosine to odd indices, shape: (max_len, embed_dim/2)
        
        self.register_buffer('pos_enc', position_enc.unsqueeze(0))  # Register the positional encoding tensor as a buffer, shape: (1, max_len, embed_dim)
        
    def forward(self, inputs):
        return inputs + self.pos_enc[:, :inputs.size(1)]  # Add positional encoding to input x, shape: (batch_size, seq_len, embed_dim)

# Position-wise Feed-Forward Networks 
class PoswiseFFN(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super(PoswiseFFN, self).__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)  # First linear layer, shape: (embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)  # Second linear layer, shape: (hidden_dim, embed_dim)
        self.relu = nn.ReLU()  # ReLU activation function

    def forward(self, inputs):
        return self.fc2(self.relu(self.fc1(inputs)))  # Apply first linear layer, ReLU, and second linear layer sequentially, shape: (batch_size, seq_len, embed_dim)

# Multi-Head Attention
class MultiHeadAttn(nn.Module):
    def __init__(self, embed_dim, n_heads):
        super(MultiHeadAttn, self).__init__()
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
        
        self.embed_dim = embed_dim  # Store the embedding dimension
        self.n_heads = n_heads  # Store the number of heads
        self.depth = embed_dim // n_heads  # Dimension of each head, shape: (embed_dim // n_heads)
        
        self.W_q = nn.Linear(embed_dim, embed_dim)  # Linear layer for query, shape: (embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)  # Linear layer for key, shape: (embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)  # Linear layer for value, shape: (embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)  # Linear layer for output, shape: (embed_dim, embed_dim)
        
    def scaled_dot_product_attention(self, queries, keys, values, mask=None):
        attn_scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.depth)  # Compute attention scores, shape: (batch_size, n_heads, seq_len, seq_len)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)  # Apply mask if provided
        attn_probs = torch.softmax(attn_scores, dim=-1)  # Softmax to get attention probabilities, shape: (batch_size, n_heads, seq_len, seq_len)
        output = torch.matmul(attn_probs, values)  # Compute weighted sum of values, shape: (batch_size, n_heads, seq_len, depth)
        return output
        
    def split_heads(self, x):
        batch_size, seq_len, embed_dim = x.size()  # Get input dimensions
        return x.view(batch_size, seq_len, self.n_heads, self.depth).transpose(1, 2)  # Split and transpose, shape: (batch_size, n_heads, seq_len, depth)
        
    def combine_heads(self, x):
        batch_size, _, seq_len, depth = x.size()  # Get input dimensions
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)  # Combine and reshape, shape: (batch_size, seq_len, embed_dim)
        
    def forward(self, queries, keys, values, mask=None):
        queries = self.split_heads(self.W_q(queries))  # Apply linear layer to queries and split heads, shape: (batch_size, n_heads, seq_len, depth)
        keys = self.split_heads(self.W_k(keys))  # Apply linear layer to keys and split heads, shape: (batch_size, n_heads, seq_len, depth)
        values = self.split_heads(self.W_v(values))  # Apply linear layer to values and split heads, shape: (batch_size, n_heads, seq_len, depth)
        
        attn_output = self.scaled_dot_product_attention(queries, keys, values, mask)  # Compute attention output, shape: (batch_size, n_heads, seq_len, depth)
        output = self.W_o(self.combine_heads(attn_output))  # Combine heads and apply final linear layer, shape: (batch_size, seq_len, embed_dim)
        return output

# Feed-Forward Network
class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_size, hidden_dim):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(embed_size, hidden_dim)  # First linear layer, shape: (embed_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_size)  # Second linear layer, shape: (hidden_dim, embed_size)
        self.relu = nn.ReLU()  # ReLU activation function

    def forward(self, inputs):
        return self.fc2(self.relu(self.fc1(inputs)))  # Apply first linear layer, ReLU, and second linear layer sequentially, shape: (batch_size, seq_len, embed_size)

# Encoder Block
class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, hidden_dim, dropout):
        super(EncoderBlock, self).__init__()
        self.self_attn = MultiHeadAttn(embed_dim, n_heads)  # Self-attention layer
        self.feed_forward = PoswiseFFN(embed_dim, hidden_dim)  # Feed-forward layer
        self.norm1 = nn.LayerNorm(embed_dim)  # Layer normalization for self-attention
        self.norm2 = nn.LayerNorm(embed_dim)  # Layer normalization for feed-forward
        self.dropout = nn.Dropout(dropout)  # Dropout layer
        
    def forward(self, inputs, mask):
        attn_output = self.self_attn(inputs, inputs, inputs, mask)  # Apply self-attention, shape: (batch_size, seq_len, embed_dim)
        inputs = self.norm1(inputs + self.dropout(attn_output))  # Apply dropout and residual connection, followed by layer norm, shape: (batch_size, seq_len, embed_dim)
        ff_output = self.feed_forward(inputs)  # Apply feed-forward network, shape: (batch_size, seq_len, embed_dim)
        inputs = self.norm2(inputs + self.dropout(ff_output))  # Apply dropout and residual connection, followed by layer norm, shape: (batch_size, seq_len, embed_dim)
        return inputs


# Decoder Block
class DecoderBlock(nn.Module):  # Define the Decoder Block class inheriting from nn.Module
    def __init__(self, embed_dim, heads, ff_dim, dropout_rate):  # Initialize with embed dimension, number of heads, feed-forward dimension, and dropout rate
        super(DecoderBlock, self).__init__()  # Call the parent constructor
        self.self_attention = MultiHeadAttn(embed_dim, heads)  # Multi-head self-attention layer
        self.enc_dec_attention = MultiHeadAttn(embed_dim, heads)  # Multi-head encoder-decoder attention layer
        self.feed_forward = PoswiseFFN(embed_dim, ff_dim)  # Position-wise feed-forward layer
        self.norm1 = nn.LayerNorm(embed_dim)  # Layer normalization after self-attention
        self.norm2 = nn.LayerNorm(embed_dim)  # Layer normalization after encoder-decoder attention
        self.norm3 = nn.LayerNorm(embed_dim)  # Layer normalization after feed-forward
        self.dropout = nn.Dropout(dropout_rate)  # Dropout layer

    def forward(self, x, enc_out, src_mask, tgt_mask):  # Forward pass method
        self_attn_out = self.self_attention(x, x, x, tgt_mask)  # Apply self-attention
        x = self.norm1(x + self.dropout(self_attn_out))  # Add, apply dropout, and normalize
        enc_dec_attn_out = self.enc_dec_attention(x, enc_out, enc_out, src_mask)  # Apply encoder-decoder attention
        x = self.norm2(x + self.dropout(enc_dec_attn_out))  # Add, apply dropout, and normalize
        ff_out = self.feed_forward(x)  # Apply feed-forward network
        x = self.norm3(x + self.dropout(ff_out))  # Add, apply dropout, and normalize
        return x  # Return the final output

# Transformer Model
class TransformerModel(nn.Module):  # Define the TransformerModel class inheriting from nn.Module
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_dim, heads, num_layers, ff_dim, max_len, dropout_rate):  # Initialize with various hyperparameters
        super(TransformerModel, self).__init__()  # Call the parent constructor
        self.enc_embedding = nn.Embedding(src_vocab_size, embed_dim)  # Embedding layer for the encoder
        self.dec_embedding = nn.Embedding(tgt_vocab_size, embed_dim)  # Embedding layer for the decoder
        self.pos_encoding = PosEnc(embed_dim, max_len)  # Positional encoding layer

        self.encoder_layers = nn.ModuleList([EncoderBlock(embed_dim, heads, ff_dim, dropout_rate) for _ in range(num_layers)])  # List of encoder layers
        self.decoder_layers = nn.ModuleList([DecoderBlock(embed_dim, heads, ff_dim, dropout_rate) for _ in range(num_layers)])  # List of decoder layers

        self.fc = nn.Linear(embed_dim, tgt_vocab_size)  # Final fully connected layer
        self.dropout = nn.Dropout(dropout_rate)  # Dropout layer

    def create_mask(self, src, tgt):  # Method to create masks
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)  # Source mask
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)  # Target mask
        seq_len = tgt.size(1)  # Get sequence length
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_len, seq_len), diagonal=1)).bool()  # No peak mask for target
        tgt_mask = tgt_mask & nopeak_mask  # Combine target mask with no peak mask
        return src_mask, tgt_mask  # Return source and target masks

    def forward(self, src, tgt):  # Forward pass method
        src_mask, tgt_mask = self.create_mask(src, tgt)  # Create masks
        src_emb = self.dropout(self.pos_encoding(self.enc_embedding(src)))  # Apply embedding, positional encoding, and dropout to source
        tgt_emb = self.dropout(self.pos_encoding(self.dec_embedding(tgt)))  # Apply embedding, positional encoding, and dropout to target

        enc_out = src_emb  # Set encoder output to source embedding initially
        for layer in self.encoder_layers:  # Iterate through encoder layers
            enc_out = layer(enc_out, src_mask)  # Apply each encoder layer

        dec_out = tgt_emb  # Set decoder output to target embedding initially
        for layer in self.decoder_layers:  # Iterate through decoder layers
            dec_out = layer(dec_out, enc_out, src_mask, tgt_mask)  # Apply each decoder layer

        output = self.fc(dec_out)  # Apply final fully connected layer
        return output  # Return the final output

if __name__ == "__main__": 
    # Define parameters
    batch_size = 2  # Batch size
    max_len = 4  # Maximum sequence length
    embed_dim = 6  # Embedding dimension
    src_vocab_size = 10  # Source vocabulary size
    tgt_vocab_size = 10  # Target vocabulary size
    num_heads = 3  # Number of attention heads
    ff_dim = 8  # Feed-forward dimension
    num_layers = 2  # Number of layers
    dropout_rate = 0.1  # Dropout rate

    model = TransformerModel(src_vocab_size, tgt_vocab_size, embed_dim, num_heads, num_layers, ff_dim, max_len, dropout_rate)  # Initialize Transformer model

    # Generate random sample data
    src_data = torch.randint(1, src_vocab_size, (batch_size, max_len))  # Random source data
    tgt_data = torch.randint(1, tgt_vocab_size, (batch_size, max_len))  # Random target data

    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Define loss function
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)  # Define optimizer

    model.train()  # Set model to training mode

    for epoch in range(5):  # Iterate through epochs
        optimizer.zero_grad()  # Zero the gradients
        output = model(src_data, tgt_data[:, :-1])  # Get model output
        loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))  # Compute loss
        loss.backward()  # Backpropagate
        optimizer.step()  # Update model parameters
        print(f"Epoch: {epoch+1}, Loss: {loss.item()}")  # Print epoch and loss