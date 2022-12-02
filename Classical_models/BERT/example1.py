import re
from functools import reduce

# def get_token_embeding(word_list, )

text = (
       'Hello, how are you? I am Romeo.\n'
       'Hello, Romeo My name is Juliet. Nice to meet you.\n'
       'Nice meet you too. How are you today?\n'
       'Great. My baseball team won the competition.\n'
       'Oh Congratulations, Juliet\n'
       'Thanks you Romeo'
   )

sentences = re.sub("[.,!?-]", '', text.lower()).split('\n') 

word_dict = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3, '[END]': 4}
special_tokens = word_dict.keys()
word_list = list(" [SEP] ".join(sentences).split())

for w in word_list:
    if w not in word_dict:
        word_dict[w] = len(word_dict)
print(word_dict)

## the first token is [CLS]
word_list.insert(0, '[CLS]')
word_list.append('[END]')
print(word_list)


## pad sentense
def sentense_padding(word_l: list) -> list:
    '''
    pad sentense to make all equal length
    this can be done in the data acquisition step also
    
    Input args:
        word_l: word list contains token [CLS], ['SEP'], [END], ['PAD'], ['MASK]
    
    Returns:
        word list
    '''
    word_len = len(word_l)
    indices = [i for i, x in enumerate(word_l) if x == "[SEP]"]
    indices.insert(0, 0)
    indices.append(word_len-int(word_l[-1] == '[END]'))
    sentences_num = len(indices)-1
    sentences_length = [indices[i+1]-indices[i]-1 for i in range(sentences_num)]  ## substract the special token
    max_length = max(sentences_length)
    print(sentences_num, sentences_length)
    ## append [PAD] in backward order so that not distroy the index 
    for i in range(0, sentences_num)[::-1]:
        word_l[indices[i+1]:indices[i+1]] = ['[PAD]']*(max_length-sentences_length[i])
    
    return word_l, max_length
    
word_list, max_length = sentense_padding(word_l=word_list)
print(word_list)
    

## create token embedding
def get_word_embedding(word_l: list, word_d: dict) -> list:
    '''
    What is token embedding?

    For instance, if the sentence is “The cat is walking. The dog is barking”, 
    then the function should create a sequence in the following manner: “[CLS] the cat is walking [SEP] the dog is barking”. 

    After that, we convert everything to an index from the word dictionary. 
    So the previous sentence would look something like “[1, 5, 7, 9, 10, 2, 5, 6, 9, 11]”. Keep in mind that 1 and 2 are [CLS] and [SEP] respectively. 
    
    Input args:
        word_l: word list contains token [CLS], ['SEP'], [END], ['PAD'], ['MASK]
        word_d: a map from token to its indice
    
    Returns:
        word embedding
    '''
    return list(map(lambda x: word_d[x], word_l))

w_e = get_word_embedding(word_l=word_list, word_d=word_dict)
print(w_e)

## create segment embedding
def get_seg_embedding(word_l: list, s_idx: int) -> list:
    '''
    A segment embedding separates two sentences from each other and they are generally defined as 0 and 1. 
    
    Input args:
        word_l: word list contains token [CLS], ['SEP'], [END], ['PAD'], ['MASK]
        s_idx: index to the sentense of interest
    
    Returns:
        segment embedding: start with token and followed by a sentence. E.g., [CLS] sentense, or [SEP] sentense
    '''
    
    ## s_idx start from 1, marked sentense format: [token] word
    w_len = len(word_l)
    indices = [i for i, x in enumerate(word_l) if x == "[SEP]"]
    indices.insert(0, 0)
    indices.append(w_len)
    
    ## prepare return list
    def map2msk(x):
        if x < indices[s_idx-1] or x >= indices[s_idx]:
            return 0
        else:
            return 1
    return list(map(map2msk, range(w_len)))

s_e = get_seg_embedding(word_l=word_list, s_idx=2)
print(s_e)

## create position embedding
def get_pos_embedding(word_l: list):
    '''
    A position embedding gives position to each embedding in a sequence. 
    
    Input args:
        word_l: word list contains token [CLS], ['SEP'], [END], ['PAD'], ['MASK]
    
    Returns:
        position embedding: a list of indices start from 0
    '''
    return list(range(len(word_l)))


p_e = get_pos_embedding(word_l=word_list)
print(p_e)
    


## build the model
import torch
import torch.nn as nn

vocab_size = len(word_dict)
maxlen = max_length
n_segments = max_length
d_model = 1000

class Embedding(nn.Module):
   def __init__(self):
       super(Embedding, self).__init__()
       self.tok_embed = nn.Embedding(vocab_size, d_model)  # token embedding
       self.pos_embed = nn.Embedding(maxlen, d_model)  # position embedding
       self.seg_embed = nn.Embedding(n_segments, d_model)  # segment(token type) embedding
       self.norm = nn.LayerNorm(d_model)

   def forward(self, x, seg):
       seq_len = x.size(1)
       pos = torch.arange(seq_len, dtype=torch.long)
       pos = pos.unsqueeze(0).expand_as(x)  # (seq_len,) -> (batch_size, seq_len)
       embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
       return self.norm(embedding)
   
def get_attn_pad_mask(seq_q, seq_k):
   batch_size, len_q = seq_q.size()
   batch_size, len_k = seq_k.size()
   # eq(zero) is PAD token
   pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k(=len_q), one is masking
   return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k

class EncoderLayer(nn.Module):
   def __init__(self):
       super(EncoderLayer, self).__init__()
       self.enc_self_attn = MultiHeadAttention()
       self.pos_ffn = PoswiseFeedForwardNet()

   def forward(self, enc_inputs, enc_self_attn_mask):
       enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
       enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size x len_q x d_model]
       return enc_outputs, attn
   


class MultiHeadAttention(nn.Module):
   def __init__(self):
       super(MultiHeadAttention, self).__init__()
       self.W_Q = nn.Linear(d_model, d_k * n_heads)
       self.W_K = nn.Linear(d_model, d_k * n_heads)
       self.W_V = nn.Linear(d_model, d_v * n_heads)

   def forward(self, Q, K, V, attn_mask):
        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size x n_heads x len_q x len_k]

        # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v) # context: [batch_size x len_q x n_heads * d_v]
        output = nn.Linear(n_heads * d_v, d_model)(context)


        return nn.LayerNorm(d_model)(output + residual), attn # output: [batch_size x len_q x d_model]
    

class ScaledDotProductAttention(nn.Module):
   def __init__(self):
       super(ScaledDotProductAttention, self).__init__()

   def forward(self, Q, K, V, attn_mask):
       scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
       scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
       attn = nn.Softmax(dim=-1)(scores)
       context = torch.matmul(attn, V)
       return score, context, attn