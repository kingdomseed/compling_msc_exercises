from audioop import bias
import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    """
    Single-head self-attention mechanism with manual QKV computation.
    Tries to represent the current token by the fused knowledge 
    of the other tokens (weighted sum). This is what it tries to learn.
    Input Shape: [batch_size, seq_len, hidden_dim]
    For self-attention we need to initialization at least three
    Linear projection modules. (in_features=green, out_features=hidden_dim)
    For forward pass 
    """
    def __init__(self, hidden_dim):
        super().__init__()
        # initialize three linear project modules
        # in the slide, it should be nn.Linear(in_features=3, out_features=3)
        self.hidden_dim = hidden_dim
        # self.query = nn.Linear(hidden_dim, hidden_dim)
        # self.key = nn.Linear(hidden_dim, hidden_dim)
        # self.value = nn.Linear(hidden_dim, hidden_dim)
        self.proj_Q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.proj_K = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.proj_V = nn.Linear(hidden_dim, hidden_dim, bias=False)

      
    """
    For forward pass we need to do dot product: Q: [batch_size, seq_len, dim_q]
    K: [batch_size, seq_len, dim_k]
    dot product between every vector in q lists and k lists. 
    Output: [batch_size, seq_len, seq_len]
    Transpose second vector--switching these two dimensions then the first
    one the shape won't change but we get the transpose of the second
    [seq_len, dim_q] x [seq_len, dim_k] -> [seq_len, seq_len]
    To understand the shape of softmax we must get the goal. 
    """
    def forward(self, x):
        # input shape: [batch_size, seq_len, hidden_dim]
        print(f'Shape: {x.shape}')
        
        # step 0: get the Q, K, V representations of the input
        Q = self.proj_Q(x)
        print(f'Query shape: {Q.shape}')
        print(f'1, 15, 128 where 15 is the number of words and 128 is the hidden dimension vector')
        
        K = self.proj_K(x)
        print(f'Key shape: {K.shape}')
        
        V = self.proj_V(x)
        print(f'Value shape: {V.shape}')
        
        # step 1: similarity matrix, Q K 
        similarity = torch.bmm(Q, K.transpose(1, 2))
        print(f'Similarity shape: {similarity.shape}')
        
        # step 2: divide that by the magic number
        similarity = similarity / math.sqrt(self.hidden_dim)
        
        # step 3: do the sofftmax
        similarity = torch.softmax(similarity, dim=-1)
        
        # step 4: get the weighted sum
        output = torch.bmm(similarity, V)
        print(f'Output shape: {output.shape}')
        # output shape: [batch_size, 15, 128]

        return output
        


class TransformerBlock(nn.Module):
    """
    Slide 28:
    A simple transformer block with:
    1. Self-attention (We don't use multi-head attention)
    2. Residual connection + Layer normalization
    3. Feed-forward network
    4. Residual connection + Layer normalization
    """
    def __init__(self, hidden_dim, ff_dim=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        # you can set it yourself, but usually it's 4xhidden dim
        self.ff_dim = ff_dim if ff_dim is not None else hidden_dim * 4

        # 1. self attention 
        self.self_attention = SelfAttention(hidden_dim)

        # 2> layernorm after the 1st sub-module 
        self.ln1 = nn.LayerNorm(hidden_dim)

        # 3. feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, self.ff_dim),
            nn.ReLU(),
            nn.Linear(self.ff_dim, hidden_dim)
        )

        # 4. layernorm after the 2nd sub-module
        self.ln2 = nn.LayerNorm(hidden_dim)
    
    def forward(self, x):
        # input x shape: [batch_size, seq_len, hidden_dim]
        # attn_output: [batch_size, seq_len, hidden_dim]
        # now we assume each token gets some information from the other tokens
        # using self-attention
        attn_output = self.self_attention(x)

        # residual connection: x + attn_output
        # layernorm: [batch_size, seq_len, hidden_dim]
        x = self.ln1(x + attn_output)
        
        # 3. feed-forward network
        # output shape: [batch_size, seq_len, hidden_dim]
        ff_output = self.ff(x)
        
        # residual connection: x + ff_output
        # layernorm: [batch_size, seq_len, hidden_dim]
        x = self.ln2(x + ff_output)
        
        return x


class Encoder(nn.Module):
    """
    Encoder using a simple transformer block instead of LSTM.
    For compatibility with the decoder, we still return (output, hidden, cell)
    where hidden and cell are derived from the transformer output.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.hidden_dim = hidden_dim
        
        # a very simple learnable positional embedding
        # in practice, it should be sinusoidal or other methods
        self.pos_embedding = nn.Parameter(torch.randn(1, 15, embedding_dim))  
        self.embedding_proj = nn.Linear(embedding_dim, hidden_dim)
        
        self.transformer_block = TransformerBlock(hidden_dim)
        
    
    def forward(self, x):
        # x: shape [batch_size, seq_len, vocab_size] one-hot embedding
        # embedded: shape [batch_size, seq_len, embedding_dim]
        embedded = self.embedding(x)  

        # add positional encoding
        embedded = embedded + self.pos_embedding 

        # project to hidden dim because 
        # our transformer block expects the input to be [batch_size, seq_len, hidden_dim]
        embedded = self.embedding_proj(embedded)  
        
        # pass through transformer block
        # output shape: [batch_size, seq_len, hidden_dim]
        output = self.transformer_block(embedded) 

        # you can ignore the following part,
        # i added them just for the compatibility with our lstm-based decoder
        # the lstm-decoder requires the hidden and cell
        # but they are not provided by the transformer block
        # so i prepared some fake hidden and cell 
        hidden_mean = output.mean(dim=1, keepdim=True)  
        hidden_mean = hidden_mean.transpose(0, 1)  
        hidden = hidden_mean
        cell = hidden_mean.clone()
        return output, hidden, cell


class Attention(nn.Module):
    """
    Two type of attention:
    1. Luong (Dot Productor) atttention: score(query, encoder_outputs) = query^T * W * encoder_outputs
    2. Bahdanau (MLP) attention: score(query, encoder_outputs) = w1 * tanh(w2[query; encoder_outputs]) 
    """
    def __init__(self, hidden_dim, type='luong'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.type = type
        if self.type == 'luong':
            self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        elif self.type == 'mlp':
            self.w1 = nn.Linear(hidden_dim * 2, hidden_dim)
            self.w2 = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, query, encoder_outputs):
        # query is the representation of the current step
        # shape is [batch_size=1, num_token=1, dim=128]
        # encoder_outputs is the representation of the original input sequence
        # shape is [batch_size=1, seq_len=15, dim=128] 
        scores = None
        if self.type == 'luong':
            # score = query^T * W * encoder_outputs
            # W*query = self.W(query); [-, 1, 128] -> [-, 1, 128]
            # bmm: [batch, n, m] * [batch, m, l] -> [batch, n, l]
            # first mat bmm:  [-, 1, 128] [batch, n, m]
            # second mat bmm: [-, 15, 128] [batch, m, l]
            # issue: we cannot use bmm directly because the 
            # matrix multiplication cannot be performend if the inner
            # dimensions are not compatible
            # solution: transpose the encoder_outputs to [-, 128, 15]
            # by using .transpose(-2, -1)
            # bmm: [-, 1, 128] x [-, 128, 15] --> [-, 1, 15]
            scores = torch.bmm(self.W(query), encoder_outputs.transpose(-2, -1))
        elif self.type == 'mlp':
            # score = w2 * tanh(w1[query; encoder_outputs])
            # repeat the query to the same length as the encoder_outputs
            # so we can concatenate them as the [h, s_k] in the slides [slide page 88]
            # after query_expanded, the shape is [-, 15, 128]
            query_expanded = query.expand(-1, encoder_outputs.size(1), -1)
            # now we have 15 tokens in the query_expanded, and 15 tokens in the encoder_outputs
            # we for each token in the encoder_outputs, we concat that with the corresponding 
            # token in the query_expanded
            # dim=-1 means concatenate the last dimension
            concat = torch.cat([query_expanded, encoder_outputs], dim=-1)
            # now the shape of concat is [batch_size=1, num_token=15, dim=128*2=256]
            # we need to project the concat vector to a different latent space using w1 
            # in slide 88, w1 is the W_1
            # --> so we do result1 = self.w1(concat), the shape of the result1 is [batch_size=1, num_token=15, dim=128]
            # then we need to apply the tanh activation function to the result1
            # --> so we do result2 = torch.tanh(self.w1(concat)), shape [batch_size=1, num_token=15, dim=128]
            # then we need to project result2 to a different latent space using w2 
            # in slide 88, w2 is the W^T_2 
            # --> so we do result3 = self.w2(result2) = self.w2(torch.tanh(self.w1(concat))), shape [batch_size=1, num_token=15, dim=1]
            # result3 is the score, [1, 15, 1], however, not done yet!
            # because when we calculate the attn_weight, that line of code will compute the softmax over the last dimension
            # to normalize the scores across the whole sequence
            # so the last dimension should mean the length of tokens, instead of 1
            # but right now, the last dimension is 1 for result3
            # --> so we do scores = result3.transpose(-2, -1) to switch the last two dimensions
            # finally, the shape of scores is [batch_size=1, num_token=1, seq_len=15]
            # scores[-, i, j] means for relation between the i-th token in the query, and the j-th token in the encoder_outputs
            scores = self.w2(torch.tanh(self.w1(concat))).transpose(-2, -1)

        # suppose the input token is '<sos> a b c <eos>'
        # and the current query token is b 
        # then the scores tensor tells us the relation between the query token b
        # and each token in the encoder_outputs '<sos> a b c <eos>'
        # and we want to represent the query token b as a weighted sum of the encoder_outputs
        # the scores tensor itself it not normalized yet, so it might look like [0, 10, 50, 40, 0]
        # if we apply softmax over the last dimension (which means the length of tokens),
        # we should get the normalized weights [0, 0.1, 0.5, 0.4, 0] 
        # the shape of attn_weights is [batch_size=1, num_token=1, seq_len=15]
        attn_weights = torch.softmax(scores, dim=-1)

        # now we get the weighted sum of the encoder_outputs
        # attn_weights is [batch_size=1, num_token=1, seq_len=15]
        # encoder_outputs is [batch_size=1, seq_len=15, dim=128]
        # we can use bmm to do apply matrix multiplication between 
        # a tensor [-, 1, 15] and a tensor [-, 15, 128]
        # the inner dimensions are compatible, both are 15
        # the output shape is then [-, 1, 128]
        # finally, we get the context vector
        # since it's the weighted sum of the encoder_outputs
        # we say the context vector encode the meaning of the source sequence
        # provide the 'context' of the task 
        context = torch.bmm(attn_weights, encoder_outputs)
        return context

class DecoderWithoutAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, hidden, cell, encoder_outputs):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc(output)
        return prediction, hidden, cell

class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)
    
    def forward(self, x, hidden, cell, encoder_outputs):
        # x is the token of the current step, shape is [1, 1] (one token at a time)
        # [batch_size=1, seq_len=1]
        embedded = self.embedding(x)
        # embedded is the embedded token, shape is [1, 1, embedding_dim=64]
        # it means we represent the current token using a 64-dim vector
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        # output is the output representation of the current step, [1, 1, embed_dim=128]
        # in lstm, hidden state and cell state have the same dimension as the embedding dimension
        # hidden state [1, 1, hidden_dim=128]
        # cell state [1, 1, hidden_dim=128]
        
        # given the current output, and all the previous history, we calculate the 
        # context using attention
        # check the code of Attention class's forward
        # it shows that the shape of context is [batch_size=1, num_token=1, dim=128]
        context = self.attention(output, encoder_outputs)

        # as introduced in the slide 89, we need to 
        # "combine source context c^{(t)} and decoder state h_t to make a prediction"
        # output is the decoder state h_t, shape is [1, 1, hidden_dim=128]
        # context is the source context c^{(t)}, shape is [1, 1, dim=128]
        # we concatenate them along the last dimension to form a new vector
        # the shape of combined is [1, 1, hidden_dim*2=256]
        combined = torch.cat([output, context], dim=-1)

        # then we "make a prediction" by projecting the combined vector to the vocabulary space
        # the shape will be [batch_size=1, num_token=1, vocab_size=29]
        # it means for this given token a, the model predicts the probability of each token in the vocabulary
        # if we choose to token that has the highest probability, suppose it's b
        # then we say given the input token a, the model thinks the most likely next token is b
        prediction = self.fc(combined)
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        self.encoder = Encoder(vocab_size, embedding_dim, hidden_dim, num_layers)
        if use_attention:
            self.decoder = Decoder(vocab_size, embedding_dim, hidden_dim, num_layers)
        else:
            self.decoder = DecoderWithoutAttention(vocab_size, embedding_dim, hidden_dim, num_layers)
    
    def forward(self, src, trg):
        # src is the input sequence (aka source sequence)
        # since all sequences are padded to the same length, shape [batch_size=1, seq_len=15]
        # trg is the target sequence
        # all target seq are also padded to the same length, shape [batch_size=1, seq_len=15]

        # we use the encoder (1-layer LSTM) to encode the input sequence
        # the encoder will output the encoder_outputs, shape is [batch_size=1, seq_len=15, dim=128]
        # each token is the input sequence is encoded into a 128-dim vector
        encoder_outputs, hidden, cell = self.encoder(src)

        # now we use decoder to predict the target sequence token by token
        decoder_outputs = []
        decoder_input = trg[:, 0:1]
        # we loop through all the tokens in the target sequence
        # trg.size(1) is the max length
        for t in range(trg.size(1)):
            # decoder_input is the token of the current step, shape is [batch_size=1, num_token=1]
            # hidden, cell are the last step from the encoder LSTM
            # they are supposed to encode the info of the whole input sequence (aka source sequence)
            # we can reinforce the source seq info by passing the encoder_outputs to the decoder
            # and then use attention mechanism to extract the most relevant information from the source seq
            # and combine that with the current decoder state to make a prediction (slide 89)
            output, hidden, cell = self.decoder(decoder_input, hidden, cell, encoder_outputs)

            # output is the prediction of the current step, shape is [batch_size=1, num_token=1, vocab_size=29]
            # it means for this given token, the model predicts the probability of each token in the vocabulary
            # we append to results to decoder_outputs
            decoder_outputs.append(output)

            # now we move to the next step
            decoder_input = trg[:, t:t+1]

        # decoder_outputs is a list of tensors, each tensor has the shape [batch_size=1, num_token=1, vocab_size=29]
        # there are 15 tokens in total
        # we concat them along the second dimension (only the value at the second dim will change)
        # the 1st dim [batch_size=1] and 3rd dim [vocab_size=29] won't change
        # the 2nd dim [num_token=1] will be changed to 15
        # so the final output shape is [batch_size=1, seq_len=15, vocab_size=29]
        return torch.cat(decoder_outputs, dim=1)
    
    def predict(self, src, max_len, sos_idx, eos_idx):
        # can be used in inference mode
        # when we want to generate the target sequence token by token
        # it's greedy search strategy, for each time step, 
        # we choose the token with the highest probability as the next token

        encoder_outputs, hidden, cell = self.encoder(src)
        decoder_input = torch.tensor([[sos_idx]]).to(src.device)
        outputs = []
        for t in range(max_len):
            output, hidden, cell = self.decoder(decoder_input, hidden, cell, encoder_outputs)
            pred = output.argmax(dim=-1)
            token_id = pred[0, 0].item()
            outputs.append(token_id)
            if token_id == eos_idx:
                break
            decoder_input = pred
        return outputs

