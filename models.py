"""Top-level model classes.

"""

import layers
import torch
import torch.nn as nn
from layers import DepthwiseSeparableConv2, PositionalEncoder
import numpy as np
import torch.nn.functional as F

        

class QANetRevised(nn.Module):
    def __init__(self, word_vectors, char_vectors, hidden_size, args, drop_prob= 0.):
        super(QANetRevised, self).__init__()
        self.emb = layers.QAEmbedding(word_vectors = word_vectors, char_vectors = char_vectors, hidden_size = hidden_size,
                                        drop_prob = drop_prob)
        self.att = layers.BiDAFAttention(hidden_size=hidden_size,
                                         drop_prob=drop_prob)
        self.context_encoder = layers.EncoderBlock(conv_layers = args.num_conv_layers,  k = args.encoder_k, hidden_dim = hidden_size, dropout_prob = drop_prob, attention_heads = args.attention_heads)
        self.query_encoder = layers.EncoderBlock(conv_layers = args.num_conv_layers, k = args.encoder_k, hidden_dim = hidden_size, dropout_prob=drop_prob, attention_heads = args.attention_heads)
        self.encoder_block_1 = layers.EncoderBlock(conv_layers = 2,  k = 5, hidden_dim = hidden_size, dropout_prob = drop_prob, attention_heads = args.attention_heads)
        self.encoder_block_2 = layers.EncoderBlock(conv_layers = 2,  k = 5, hidden_dim = hidden_size, dropout_prob = drop_prob, attention_heads = args.attention_heads)
        self.encoder_block_3 = layers.EncoderBlock(conv_layers = 2,  k = 5, hidden_dim = hidden_size, dropout_prob = drop_prob, attention_heads = args.attention_heads)
        self.stacked_encoder_blocks = nn.ModuleList([self.encoder_block_1, self.encoder_block_2, self.encoder_block_3])
        self.output = layers.QANetOutput(hidden_size = hidden_size, dropout_prob = drop_prob)
        self.resize_attn = layers.DepthwiseSeparableConv(hidden_size * 4, hidden_size, 5)
        self.hidden_size = hidden_size
        self.resize_context = DepthwiseSeparableConv2(500, self.hidden_size, 7)
        self.resize_query = DepthwiseSeparableConv2(500, self.hidden_size, 7)

    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs, args):

        
        self.context_pos_enc = self.get_sinusoid_encoding_table(cw_idxs.size(1), self.hidden_size, padding_idx= 0)
        self.query_pos_enc = self.get_sinusoid_encoding_table(qw_idxs.size(1), self.hidden_size, padding_idx= 0)
        c_mask = self.get_attn_key_pad_mask(cw_idxs, cw_idxs)
        c_non_pad_mask = self.get_non_pad_mask(cw_idxs)
        q_mask = self.get_attn_key_pad_mask(qw_idxs, qw_idxs)
        q_non_pad_mask = self.get_non_pad_mask(qw_idxs)

        # Embed Context
        c_emb = self.emb(cw_idxs, cc_idxs)         # (batch_size, c_len, hidden_size)
        c_emb = self.resize_context(c_emb.permute(0,2,1))
        c_emb = F.relu(c_emb)
        c_emb = c_emb.permute(0,2,1)
        c_emb = c_emb + self.context_pos_enc

        # Embed Query
        q_emb = self.emb(qw_idxs, qc_idxs)         # (batch_size, q_len, hidden_size) 
        q_emb = self.resize_query(q_emb.permute(0,2,1))
        q_emb = F.relu(q_emb)
        q_emb = q_emb.permute(0,2,1)
        q_emb = q_emb + self.query_pos_enc

        c_enc = self.context_encoder(c_emb, c_mask, c_non_pad_mask, cw_idxs.size(1), args.hidden_size, args.num_conv_layers)
        q_enc = self.query_encoder(q_emb, q_mask, q_non_pad_mask, qw_idxs.size(1), args.hidden_size, args.num_conv_layers)
        
        # Using BiDAF Attention over context and query encodings
        c_mask_2 = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask_2 = torch.zeros_like(qw_idxs) != qw_idxs
        att = self.att(c_enc, q_enc, c_mask_2, q_mask_2)
        att = att.permute(0,2,1)
        enc_1 = self.resize_attn(att)
        enc_1 = enc_1.permute(0,2,1)
        
        for encoder in self.stacked_encoder_blocks:
            enc_1 = enc_1 + self.context_pos_enc
            enc_1 = encoder(enc_1, c_mask, c_non_pad_mask, cw_idxs.size(1), args.hidden_size, args.num_conv_layers)
        enc_2 = enc_1
        for encoder in self.stacked_encoder_blocks:
            enc_2 = enc_2 + self.context_pos_enc
            enc_2 = encoder(enc_2, c_mask, c_non_pad_mask, cw_idxs.size(1), args.hidden_size, args.num_conv_layers)
        enc_3 = enc_2
        for encoder in self.stacked_encoder_blocks:
            enc_3 = enc_3 + self.context_pos_enc
            enc_3 = encoder(enc_3, c_mask, c_non_pad_mask, cw_idxs.size(1), args.hidden_size, args.num_conv_layers)
        p1, p2 = self.output(enc_1, enc_2, enc_3, c_mask_2)

        return p1,p2



    def get_non_pad_mask(self,seq):
        assert seq.dim() == 2
        return seq.ne(0).type(torch.float).unsqueeze(-1)


    def get_attn_key_pad_mask(self,seq_k, seq_q):
        ''' For masking out the padding part of key sequence. '''
        # Expand to fit the shape of key query attention matrix.
        len_q = seq_q.size(1)
        padding_mask = seq_k.eq(0)
        padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
        return padding_mask



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




class BiDAF(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob=0., args = None):
        super(BiDAF, self).__init__()
        self.emb = layers.QAEmbedding(word_vectors=word_vectors, char_vectors = char_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)
                                
        self.context_enc = layers.EncoderBlock(conv_layers = args.num_conv_layers,  k = args.encoder_k, hidden_dim = hidden_size, dropout_prob = drop_prob, attention_heads = args.attention_heads)
        self.question_enc = layers.EncoderBlock(conv_layers = args.num_conv_layers, k = args.encoder_k, hidden_dim = hidden_size, dropout_prob=drop_prob, attention_heads = args.attention_heads)
        self.context_conv = DepthwiseSeparableConv2(500,hidden_size, 5)
        self.question_conv = DepthwiseSeparableConv2(500,hidden_size, 5)
        self.att = layers.BiDAFAttention(hidden_size=hidden_size,
                                         drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)
        self.mod_enc = layers.EncoderBlock(conv_layers = 2,  k = 5, hidden_dim = hidden_size, dropout_prob = drop_prob, attention_heads = args.attention_heads)
        self.cq_resizer = DepthwiseSeparableConv2(hidden_size * 4, hidden_size, 5)
        self.out = layers.BiDAFOutput2(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs, args):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_mask_2 = (torch.zeros_like(cw_idxs) == cw_idxs).float()
        q_mask_2 = (torch.zeros_like(qw_idxs) == qw_idxs).float()
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs, cc_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs, qc_idxs)         # (batch_size, q_len, hidden_size)
        c_emb = c_emb.permute(0,2,1)
        q_emb = q_emb.permute(0,2,1)
        C = self.context_conv(c_emb)  
        Q = self.question_conv(q_emb) 

        c_enc = self.context_enc(C, c_mask_2, cw_idxs.size(1))    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.question_enc(Q, q_mask_2, qw_idxs.size(1))    # (batch_size, q_len, 2 * hidden_size)
        
        c_enc = c_enc.permute(0,2,1)
        q_enc = q_enc.permute(0,2,1)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        att = att.permute(0,2,1)
        att = self.cq_resizer(att)
        
        mod = self.mod_enc(att, c_mask_2, cw_idxs.size(1))        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out
