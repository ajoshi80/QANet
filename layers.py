"""Assortment of layers for use in models.py.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax
import numpy as np
import random


class CNN(nn.Module):
    def __init__(self, k, in_channels, out_channels):
        super(CNN, self).__init__()
        self.k = k
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv1d(in_channels, out_channels, k)

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool1d(x,x.shape[-1])
        
        return torch.squeeze(x)

class QAEmbedding(nn.Module):
    """Embedding layer.

    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """
    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob):
        super(QAEmbedding, self).__init__()
        self.drop_prob = drop_prob
        self.word_embed = nn.Embedding.from_pretrained(word_vectors, freeze = True)
        self.char_embed = nn.Embedding.from_pretrained(char_vectors, freeze = False)
        self.CNN = CNN(5, self.char_embed.embedding_dim, 200)
        self.proj = nn.Linear(word_vectors.size(1) + char_vectors.size(1), word_vectors.size(1) + char_vectors.size(1), bias=False)
        self.hwy = HighwayEncoder(2, 500)

    def forward(self, w_idx, c_idx):
        char_embeddings = self.char_embed(c_idx)
        char_embeddings = F.dropout(char_embeddings, self.drop_prob, self.training)
        og_shape = char_embeddings.shape
        char_embeddings = char_embeddings.permute(0,1,3,2)  #(batch_size, seq_len, char_embed_dim from GloVe)
        char_embeddings_reshaped = char_embeddings.view(char_embeddings.shape[0] * char_embeddings.shape[1], char_embeddings.shape[2], char_embeddings.shape[3])
        char_embeddings = self.CNN(char_embeddings_reshaped)
        char_embeddings = char_embeddings.view(og_shape[0], og_shape[1], 200)
        word_embeddings = self.word_embed(w_idx)   # (batch_size, seq_len, embed_size)
        word_embeddings = F.dropout(word_embeddings, self.drop_prob, self.training)
        emb = torch.cat([char_embeddings, word_embeddings], dim=2)
        emb = self.hwy(emb)   # (batch_size, seq_len, hidden_size)
        return emb


class HighwayEncoder(nn.Module):
    """Encode an input sequence using a highway network.

    Based on the paper:
    "Highway Networks"
    by Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
    (https://arxiv.org/abs/1505.00387).

    Args:
        num_layers (int): Number of layers in the highway encoder.
        hidden_size (int): Size of hidden activations.
    """
    def __init__(self, num_layers, hidden_size):
        super(HighwayEncoder, self).__init__()
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                    for _ in range(num_layers)])

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x

        return x


class RNNEncoder(nn.Module):
    """General-purpose layer for encoding a sequence using a bidirectional RNN.

    Encoded output is the RNN's hidden state at each position, which
    has shape `(batch_size, seq_len, hidden_size * 2)`.

    Args:
        input_size (int): Size of a single timestep in the input.
        hidden_size (int): Size of the RNN hidden state.
        num_layers (int): Number of layers of RNN cells to use.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 drop_prob=0.):
        super(RNNEncoder, self).__init__()
        self.drop_prob = drop_prob
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True,
                           bidirectional=True,
                           dropout=drop_prob if num_layers > 1 else 0.)

    def forward(self, x, lengths):
        # Save original padded length for use by pad_packed_sequence
        orig_len = x.size(1)

        # Sort by length and pack sequence for RNN
        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]     # (batch_size, seq_len, input_size)
        x = pack_padded_sequence(x, lengths, batch_first=True)

        # Apply RNN
        x, _ = self.rnn(x)  # (batch_size, seq_len, 2 * hidden_size)

        # Unpack and reverse sort
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=orig_len)
        _, unsort_idx = sort_idx.sort(0)
        x = x[unsort_idx]   # (batch_size, seq_len, 2 * hidden_size)

        # Apply dropout (RNN applies dropout after all but the last layer)
        x = F.dropout(x, self.drop_prob, self.training)

        return x


class BiDAFAttention(nn.Module):
    """Bidirectional attention originally used by BiDAF.

    Bidirectional attention computes attention in two directions:
    The context attends to the query and the query attends to the context.
    The output of this layer is the concatenation of [context, c2q_attention,
    context * c2q_attention, context * q2c_attention]. This concatenation allows
    the attention vector at each timestep, along with the embeddings from
    previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, context_len, 8 * hidden_size).

    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob=0.1):
        super(BiDAFAttention, self).__init__()
        self.drop_prob = drop_prob
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        s = self.get_similarity_matrix(c, q)        # (batch_size, c_len, q_len)
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)       # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim=1)       # (batch_size, c_len, q_len)
        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, q)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)
        x = torch.cat([c, a, c * a, c * b], dim=2)  # (bs, c_len, 4 * hid_size)

        return x

    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).

        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.

        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        c_len, q_len = c.size(1), q.size(1)
        c = F.dropout(c, self.drop_prob, self.training)  # (bs, c_len, hid_size)
        q = F.dropout(q, self.drop_prob, self.training)  # (bs, q_len, hid_size)
        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2)\
                                           .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s


class BiDAFOutput(nn.Module):
    """Output layer used by BiDAF for question answering.

    Computes a linear transformation of the attention and modeling
    outputs, then takes the softmax of the result to get the start pointer.
    A bidirectional LSTM is then applied the modeling output to produce `mod_2`.
    A second linear+softmax of the attention output and `mod_2` is used
    to get the end pointer.

    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob):
        super(BiDAFOutput, self).__init__()
        self.att_linear_1 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_1 = nn.Linear(2 * hidden_size, 1)

        self.rnn = RNNEncoder(input_size=2 * hidden_size,
                              hidden_size=hidden_size,
                              num_layers=1,
                              drop_prob=drop_prob)

        self.att_linear_2 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_2 = nn.Linear(2 * hidden_size, 1)

    def forward(self, att, mod, mask):
        # Shapes: (batch_size, seq_len, 1)
        logits_1 = self.att_linear_1(att) + self.mod_linear_1(mod)
        mod_2 = self.rnn(mod, mask.sum(-1))
        logits_2 = self.att_linear_2(att) + self.mod_linear_2(mod_2)

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2


class BiDAFOutput2(nn.Module):
    """Output layer used by BiDAF for question answering.

    Computes a linear transformation of the attention and modeling
    outputs, then takes the softmax of the result to get the start pointer.
    A bidirectional LSTM is then applied the modeling output to produce `mod_2`.
    A second linear+softmax of the attention output and `mod_2` is used
    to get the end pointer.

    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob):
        super(BiDAFOutput2, self).__init__()
        self.att_linear_1 = nn.Linear(hidden_size, 1)
        self.mod_linear_1 = nn.Linear(hidden_size, 1)

        self.rnn = RNNEncoder(input_size=hidden_size,
                              hidden_size=hidden_size,
                              num_layers=1,
                              drop_prob=drop_prob)

        self.att_linear_2 = nn.Linear( hidden_size, 1)
        self.mod_linear_2 = nn.Linear( 2*hidden_size, 1)

    def forward(self, att, mod, mask):
        # Shapes: (batch_size, seq_len, 1)
        logits_1 = self.att_linear_1(att.permute(0,2,1)) + self.mod_linear_1(mod.permute(0,2,1))
        mod_2 = self.rnn(mod.permute(0,2,1), mask.sum(-1))

        a = self.mod_linear_2(mod_2)

        logits_2 = self.att_linear_2(att.permute(0,2,1)) + self.mod_linear_2(mod_2)

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2


class QANetOutput(nn.Module):
    def __init__(self, hidden_size, dropout_prob):
        super(QANetOutput, self).__init__()
        self.linear_1 = nn.Linear(hidden_size * 2, 1)
        self.linear_2 = nn.Linear(hidden_size * 2, 1)
    
    def forward(self, M1, M2, M3, mask):
        X1 = torch.cat([M1, M2], dim=2)
        X2 = torch.cat([M1, M3], dim=2)
        logits_1 = self.linear_1(X1)
        logits_2 = self.linear_2(X2)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)
        return log_p1, log_p2


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, k, bias = True):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=k, groups=in_channels, bias= bias, padding=k//2)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias = bias, padding= 0)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -1e30)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module 
        Implementation of multihead attention referenced from https://github.com/jadore801120/attention-is-all-you-need-pytorch/tree/master/transformer
    '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.view(q.shape[0]*q.shape[2], len_q, d_k) # (n*b) x lq x dk
        k = k.view(k.shape[0]*k.shape[2], len_k, d_k) # (n*b) x lk x dk
        v = v.view(v.shape[0]*v.shape[2], len_v, d_v) # (n*b) x lv x dv
        mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)
        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn



class PositionalEncoder(nn.Module):
    def __init__(self, length, hidden_dim, padding_idx = None):
        super().__init__()
        self.positional_enc = nn.Embedding.from_pretrained(self.get_sinusoid_encoding_table(length, hidden_dim, padding_idx = padding_idx), freeze = True)

    def forward(self, x):
        return x + self.positional_enc

    #from https://github.com/jadore801120/attention-is-all-you-need-pytorch/tree/master/transformer
    def get_sinusoid_encoding_table(self, n_position, d_hid, padding_idx=None):
        
        def cal_angle(position, hid_idx):
            return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

        def get_posi_angle_vec(position):
            return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        if padding_idx is not None:
            # zero vector for padding dimension
            sinusoid_table[padding_idx] = 0.

        return torch.cuda.FloatTensor(sinusoid_table)
    


class EncoderBlock(nn.Module):
    def __init__(self, conv_layers, k, hidden_dim, dropout_prob, attention_heads):
        super(EncoderBlock,self).__init__()
        
        self.conv = nn.ModuleList([DepthwiseSeparableConv2(hidden_dim, hidden_dim, k) for _ in range(conv_layers)])
        d_k = hidden_dim//attention_heads
        d_v = hidden_dim//attention_heads
        self.self_attn = MultiHeadAttention(attention_heads , hidden_dim, d_k, d_v, dropout=dropout_prob)
        self.ff = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        # Feed Forward - > Two linear layers with relu in between. Maintain same input and output dim
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(conv_layers)])
        self.single_layer_norm_1 = nn.LayerNorm(hidden_dim)
        self.single_layer_norm_2 = nn.LayerNorm(hidden_dim)
        self.dropout_prob = dropout_prob
        self.num_conv_layers = conv_layers

            
    def forward(self, x, attn_mask, non_pad_mask, length, hidden_dim, conv_layers):

        residual = x
        for i, (convolution, layer_norm) in enumerate(zip(self.conv, self.layer_norms)):
            #
            rand = random.random()
            dropout = self.dropout_prob * (i + 1)/self.num_conv_layers
            if rand > dropout  or not self.training:
                x = layer_norm(x)
                x = x.permute(0,2,1)
                x = convolution(x)
                x = F.relu(x)
                x = x.permute(0,2,1)
                x = x + residual
                residual = x
            
        x = self.single_layer_norm_1(x)
        x, attn = self.self_attn(x, x, x, mask = attn_mask)
        x *= non_pad_mask
        x = x + residual
        residual = x
        x = self.single_layer_norm_2(x)
        x = self.ff(x)
        x = x + residual

        return x


class Embedding(nn.Module):
    """Embedding layer used by BiDAF, without the character-level component.

    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """
    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob):
        super(Embedding, self).__init__()
        self.drop_prob = drop_prob
        self.word_embed = nn.Embedding.from_pretrained(word_vectors)
        self.char_embed = nn.Embedding.from_pretrained(char_vectors)
        self.CNN = CNN(5, self.char_embed.embedding_dim, self.char_embed.embedding_dim)
        self.proj = nn.Linear(word_vectors.size(1) + char_vectors.size(1), hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size)

    def forward(self, w_idx, c_idx):
        char_embeddings = self.char_embed(c_idx)
        og_shape = char_embeddings.shape
        char_embeddings = char_embeddings.permute(0,1,3,2)
        char_embeddings_reshaped = char_embeddings.view(char_embeddings.shape[0] * char_embeddings.shape[1], char_embeddings.shape[2], char_embeddings.shape[3])
        char_embeddings = self.CNN(char_embeddings_reshaped)
        char_embeddings = char_embeddings.view(og_shape[0], og_shape[1], og_shape[3])
        word_embeddings = self.word_embed(w_idx)   # (batch_size, seq_len, embed_size)
        word_embeddings = F.dropout(word_embeddings, self.drop_prob, self.training)
        emb = torch.cat([char_embeddings, word_embeddings], dim=2)
        emb = self.proj(emb)  # (batch_size, seq_len, hidden_size)
        emb = self.hwy(emb)   # (batch_size, seq_len, hidden_size)
        return emb



class DepthwiseSeparableConv2(nn.Module):
    def __init__(self, in_ch, out_ch, k, dim=1, bias=True):
        super().__init__()
        if dim == 1:
            self.depthwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch,
                                            padding=k // 2, bias=bias)
            self.pointwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, bias=bias)
        elif dim == 2:
            self.depthwise_conv = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch,
                                            padding=k // 2, bias=bias)
            self.pointwise_conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, bias=bias)
        else:
            raise Exception("Wrong dimension for Depthwise Separable Convolution!")
        nn.init.kaiming_normal_(self.depthwise_conv.weight, nonlinearity="relu")
        nn.init.constant_(self.depthwise_conv.bias, 0.0)
        nn.init.kaiming_normal_(self.depthwise_conv.weight, nonlinearity="relu")
        nn.init.constant_(self.pointwise_conv.bias, 0.0)

    def forward(self, x):
        return self.pointwise_conv(self.depthwise_conv(x))