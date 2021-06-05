import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math

try:
    from transformers.modeling_bert import BertConfig, BertEncoder, BertModel
except:
    from transformers.models.bert.modeling_bert import BertConfig, BertEncoder, BertModel

def init_layers(args):
    cate_embedding_layers, cont_embedding_layers = {}, {}

    for cate_col in args.cate_cols:
        cate_embedding_layers[cate_col] = \
            nn.Sequential(
                nn.Embedding(args.cate_len[cate_col] + 1, args.hidden_dim//3).to(args.device) if cate_col != 'answerCode' \
                else nn.Embedding(args.cate_len[cate_col], args.hidden_dim//3).to(args.device),
                nn.LayerNorm(args.hidden_dim//3).to(args.device))

    for cont_col in args.cont_cols:
        cont_embedding_layers[cont_col] = \
            nn.Sequential(
                nn.Linear(1, args.hidden_dim//3).to(args.device),
                nn.LayerNorm(args.hidden_dim//3).to(args.device))

    # embedding combination projection
    comb_proj = nn.Sequential(
        nn.Linear((args.hidden_dim//3)*((len(args.cont_cols) + len(args.cate_cols))), args.hidden_dim),
        nn.LayerNorm(args.hidden_dim).to(args.device))

    return cate_embedding_layers, cont_embedding_layers, comb_proj


def forward_layers(args, input, cate_embedding_layers, cont_embedding_layers):

    '''
    input 순서는 category + continuous + mask
    'answerCode', 'interaction', 'assessmentItemID', 'testId', 'KnowledgeTag', + 추가 category
    추가 cont
    'mask'
    원래 코드
    test, question, tag, _, mask, interaction = input
    '''

    cate_inputs = input[1:len(args.cate_cols)+1]
    cont_inputs = input[len(args.cate_cols)+1:len(args.cate_cols)+len(args.cont_cols)+1]

    '''
    cate_embedding_layers에 input(data) 넣어서 feature로 output
    원래 코드
    embed_interaction = self.embedding_interaction(interaction)
    embed_test = self.embedding_test(test)
    embed_question = self.embedding_question(question)
    embed_tag = self.embedding_tag(tag)
    '''

    embed_cate_features = torch.cat(
            [cate_embedding_layers[cate_col](cate_input) for cate_input, cate_col \
                in zip(cate_inputs, args.cate_cols)], 2)

    if len(args.cont_cols) != 0:
        embed_cont_features = torch.cat(
                [cont_embedding_layers[cont_col](cont_input.unsqueeze(2)) for cont_input, cont_col \
                    in zip(cont_inputs, args.cont_cols)], 2)
    else:
        embed_cont_features = torch.tensor([]).to(args.device)

    embed = torch.cat([embed_cate_features,
                       embed_cont_features], 2)

    return embed


class LSTM(nn.Module):
    def __init__(self, args):
        super(LSTM, self).__init__()
        self.args = args
        self.device = args.device

        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers

        '''
        cate_embedding_layers에 category embedding layer들이 list로 저장
        cate_len dictionary에 key : cate_column의 이름, value : category의 개수
        원래 코드
        interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim//3)
        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim//3)
        self.embedding_question = nn.Embedding(self.args.n_questions + 1, self.hidden_dim//3)
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim//3)
        '''
        # Embedding

        self.cate_embedding_layers, self.cont_embedding_layers, self.comb_proj = init_layers(args)

        self.lstm = nn.LSTM(self.hidden_dim,
                            self.hidden_dim,
                            self.n_layers,
                            batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)

        self.activation = nn.Sigmoid()

    def init_hidden(self, batch_size):
        h = torch.zeros(
            self.n_layers,
            batch_size,
            self.hidden_dim)
        h = h.to(self.device)

        c = torch.zeros(
            self.n_layers,
            batch_size,
            self.hidden_dim)
        c = c.to(self.device)

        return (h, c)

    def forward(self, input):
        batch_size = input[1].size(0)
        mask = input[-1]

        # Embedding
        embed = forward_layers(self.args, input, self.cate_embedding_layers, self.cont_embedding_layers)

        X = self.comb_proj(embed)

        hidden = self.init_hidden(batch_size)
        out, hidden = self.lstm(X, hidden)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)

        out = self.fc(out)
        preds = self.activation(out).view(batch_size, -1)

        return preds

class RNNATTN(nn.Module):
    def __init__(self, args):
        super(RNNATTN, self).__init__()
        self.args = args
        self.device = args.device

        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers
        self.n_heads = self.args.n_heads
        self.drop_out = self.args.drop_out

        '''
        cate_embedding_layers에 category embedding layer들이 list로 저장
        cate_len dictionary에 key : cate_column의 이름, value : category의 개수
        원래 코드
        interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim//3)
        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim//3)
        self.embedding_question = nn.Embedding(self.args.n_questions + 1, self.hidden_dim//3)
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim//3)
        '''

        self.cate_embedding_layers, self.cont_embedding_layers, self.comb_proj = init_layers(args)

        if self.args.model == "lstmattn":
            self.rnn = nn.LSTM(self.hidden_dim,
                                self.hidden_dim,
                                self.n_layers,
                                batch_first=True)

        elif self.args.model == "gruattn":
            self.rnn = nn.GRU(self.hidden_dim,
                            self.hidden_dim,
                            self.n_layers,
                            batch_first=True)

        self.config = BertConfig(
            3, # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=1,
            num_attention_heads=self.n_heads,
            intermediate_size=self.hidden_dim,
            hidden_dropout_prob=self.drop_out,
            attention_probs_dropout_prob=self.drop_out,
        )
        self.attn = BertEncoder(self.config)

        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)

        self.activation = nn.Sigmoid()

    def init_hidden(self, batch_size):
        h = torch.zeros(
            self.n_layers,
            batch_size,
            self.hidden_dim)
        h = h.to(self.device)

        # GRU does not require cell state
        if self.args.model == "gruattn":
            return h

        c = torch.zeros(
            self.n_layers,
            batch_size,
            self.hidden_dim)
        c = c.to(self.device)

        return (h, c)

    def forward(self, input):
        batch_size = input[1].size(0)
        mask = input[-1]

        # Embedding
        embed = forward_layers(self.args, input, self.cate_embedding_layers, self.cont_embedding_layers)

        X = self.comb_proj(embed)

        hidden = self.init_hidden(batch_size)
        out, hidden = self.rnn(X, hidden)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)

        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.n_layers

        encoded_layers = self.attn(out, extended_attention_mask, head_mask=head_mask)
        sequence_output = encoded_layers[-1]

        out = self.fc(sequence_output)

        preds = self.activation(out).view(batch_size, -1)

        return preds


class Bert(nn.Module):
    def __init__(self, args):
        super(Bert, self).__init__()
        self.args = args
        self.device = args.device

        # Defining some parameters
        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers

        # Embedding
        self.cate_embedding_layers, self.cont_embedding_layers, self.comb_proj = init_layers(args)

        # Bert config
        self.config = BertConfig(
            3, # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=self.args.n_layers,
            num_attention_heads=self.args.n_heads,
            max_position_embeddings=self.args.max_seq_len
        )

        # Defining the layers
        # Bert Layer
        self.encoder = BertModel(self.config)

        # Fully connected layer
        self.fc = nn.Linear(self.args.hidden_dim, 1)

        self.activation = nn.Sigmoid()


    def forward(self, input):
        batch_size = input[1].size(0)
        mask = input[-1]

        # Embedding
        embed = forward_layers(self.args, input, self.cate_embedding_layers, self.cont_embedding_layers)

        X = self.comb_proj(embed)

        # Bert
        encoded_layers = self.encoder(inputs_embeds=X, attention_mask=mask)
        out = encoded_layers[0]
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out)
        preds = self.activation(out).view(batch_size, -1)

        return preds

class Feed_Forward_block(nn.Module):
    """
    out =  Relu( M_out*w1 + b1) *w2 + b2
    """
    def __init__(self, dim_ff):
        super().__init__()
        self.layer1 = nn.Linear(in_features=dim_ff, out_features=dim_ff)
        self.layer2 = nn.Linear(in_features=dim_ff, out_features=dim_ff)

    def forward(self,ffn_in):
        return self.layer2(F.relu(self.layer1(ffn_in)))

class LastQuery(nn.Module):
    def __init__(self, args):
        super(LastQuery, self).__init__()
        self.args = args
        self.device = args.device

        self.hidden_dim = self.args.hidden_dim

        '''
        # Embedding
        # interaction은 현재 correct으로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim//3)
        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim//3)
        self.embedding_question = nn.Embedding(self.args.n_questions + 1, self.hidden_dim//3)
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim//3)
        self.embedding_position = nn.Embedding(self.args.max_seq_len, self.hidden_dim)
        # embedding combination projection
        self.comb_proj = nn.Linear((self.hidden_dim//3)*4, self.hidden_dim)
        # 기존 keetar님 솔루션에서는 Positional Embedding은 사용되지 않습니다
        # 하지만 사용 여부는 자유롭게 결정해주세요 :)
        # self.embedding_position = nn.Embedding(self.args.max_seq_len, self.hidden_dim)
        '''
        self.cate_embedding_layers, self.cont_embedding_layers, self.comb_proj = init_layers(args)

        # Encoder
        self.query = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.key = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.value = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)

        self.attn = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=self.args.n_heads)
        self.mask = None # last query에서는 필요가 없지만 수정을 고려하여서 넣어둠
        self.ffn = Feed_Forward_block(self.hidden_dim)

        self.ln1 = nn.LayerNorm(self.hidden_dim)
        self.ln2 = nn.LayerNorm(self.hidden_dim)

        # LSTM
        self.lstm = nn.LSTM(
            self.hidden_dim,
            self.hidden_dim,
            self.args.n_layers,
            batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)

        self.activation = nn.Sigmoid()

    def get_pos(self, seq_len):
        # use sine positional embeddinds
        return torch.arange(seq_len).unsqueeze(0)

    def init_hidden(self, batch_size):
        h = torch.zeros(
            self.args.n_layers,
            batch_size,
            self.args.hidden_dim)
        h = h.to(self.device)

        c = torch.zeros(
            self.args.n_layers,
            batch_size,
            self.args.hidden_dim)
        c = c.to(self.device)

        return (h, c)


    def forward(self, input):
        '''
        test, question, tag, _, mask, interaction, index = input
        batch_size = interaction.size(0)
        seq_len = interaction.size(1)
        '''
        batch_size = input[1].size(0)
        mask = input[-1]

        '''
        # 신나는 embedding
        embed_interaction = self.embedding_interaction(interaction)
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)
        embed = torch.cat([embed_interaction,
                           embed_test,
                           embed_question,
                           embed_tag,], 2)
        '''

        embed = forward_layers(self.args, input, self.cate_embedding_layers, self.cont_embedding_layers)
        embed = self.comb_proj(embed)

        # Positional Embedding
        # last query에서는 positional embedding을 하지 않음
        # position = self.get_pos(seq_len).to('cuda')
        # embed_pos = self.embedding_position(position)
        # embed = embed + embed_pos

        ####################### ENCODER #####################
        q = self.query(embed)[:, -1:, :].permute(1, 0, 2)
        k = self.key(embed).permute(1, 0, 2)
        v = self.value(embed).permute(1, 0, 2)

        ## attention
        # last query only
        out, _ = self.attn(q, k, v)

        ## residual + layer norm
        out = out.permute(1, 0, 2)
        out = embed + out
        out = self.ln1(out)

        ## feed forward network
        out = self.ffn(out)

        ## residual + layer norm
        out = embed + out
        out = self.ln2(out)

        ###################### LSTM #####################
        hidden = self.init_hidden(batch_size)
        out, hidden = self.lstm(out, hidden)

        ###################### DNN #####################
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out)

        preds = self.activation(out).view(batch_size, -1)

        # print(preds)

        return preds