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
    cate_embedding_layers, num_embedding_layers = {}, {}

    for cate_col in args.cate_cols:
        cate_embedding_layers[cate_col] = \
            nn.Embedding(args.cate_len[cate_col] + 1, args.hidden_dim//3).to(args.device) if cate_col != 'answerCode' \
            else nn.Embedding(args.cate_len[cate_col], args.hidden_dim//3).to(args.device)

    for num_col in args.num_cols:
        num_embedding_layers[num_col] = nn.Linear(1, args.hidden_dim//3).to(args.device)

    # embedding combination projection
    comb_proj = nn.Linear((args.hidden_dim//3)*((len(args.num_cols) + len(args.cate_cols))), args.hidden_dim)

    return cate_embedding_layers, num_embedding_layers, comb_proj


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

        self.cate_embedding_layers, self.num_embedding_layers, self.comb_proj = init_layers(args)

        # for cate_col in self.args.cate_cols:
        #     self.cate_embedding_layers[cate_col] = \
        #         nn.Embedding(self.args.cate_len[cate_col] + 1, self.hidden_dim//3).to(args.device) if cate_col != 'answerCode' \
        #         else nn.Embedding(self.args.cate_len[cate_col], self.hidden_dim//3).to(args.device)


        # self.num_embedding_layers = {}
        # for num_col in self.args.num_cols:
        #     self.num_embedding_layers[num_col] = \
        #         nn.Linear(1, self.hidden_dim//3).to(args.device)

        # self.comb_proj = nn.Linear((self.hidden_dim//3)*((len(self.args.num_cols) + len(self.args.cate_cols))), self.hidden_dim)

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
        '''
        input 순서는 category + numeric + mask

        'answerCode', 'interaction', 'assessmentItemID', 'testId', 'KnowledgeTag', + 추가 category
        추가 num
        'mask'

        원래 코드
        test, question, tag, _, mask, interaction = input
        '''

        batch_size = input[1].size(0)
        cate_inputs = input[1:len(self.args.cate_cols)+1]
        num_inputs = input[len(self.args.cate_cols)+1:len(self.args.cate_cols)+len(self.args.num_cols)+1]
        
        '''
        cate_embedding_layers에 input(data) 넣어서 feature로 output

        원래 코드
        embed_interaction = self.embedding_interaction(interaction)
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)
        '''
        
        # Embedding
        embed_cate_features = torch.cat(
                [self.cate_embedding_layers[cate_col](cate_input) for cate_input, cate_col \
                    in zip(cate_inputs, self.args.cate_cols)], 2)
    
        embed_num_features = torch.cat(
                [self.num_embedding_layers[num_col](num_input.unsqueeze(2)) for num_input, num_col \
                    in zip(num_inputs, self.args.num_cols)], 2)

        embed = torch.cat([embed_cate_features,
                           embed_num_features], 2)

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

        self.cate_embedding_layers, self.num_embedding_layers, self.comb_proj = init_layers(args)

        # self.cate_embedding_layers = {}
        # for cate_col in self.args.cate_cols:
        #     self.cate_embedding_layers[cate_col] = \
        #         nn.Embedding(self.args.cate_len[cate_col] + 1, self.hidden_dim//3).to(args.device) if cate_col != 'answerCode' \
        #         else nn.Embedding(self.args.cate_len[cate_col], self.hidden_dim//3).to(args.device)


        # self.num_embedding_layers = {}
        # for num_col in self.args.num_cols:
        #     self.num_embedding_layers[num_col] = \
        #         nn.Linear(1, self.hidden_dim//3).to(args.device)

        # self.comb_proj = nn.Linear((self.hidden_dim//3)*((len(self.args.num_cols) + len(self.args.cate_cols))), self.hidden_dim)

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
        '''
        input 순서는 category + numeric + mask

        'answerCode', 'interaction', 'assessmentItemID', 'testId', 'KnowledgeTag', + 추가 category
        추가 num
        'mask'

        원래 코드
        test, question, tag, _, mask, interaction = input
        '''
        
        batch_size = input[1].size(0)
        cate_inputs = input[1:len(self.args.cate_cols)+1]
        num_inputs = input[len(self.args.cate_cols)+1:len(self.args.cate_cols)+len(self.args.num_cols)+1]
        mask = input[-1]
        
        '''
        cate_embedding_layers에 input(data) 넣어서 feature로 output

        원래 코드
        embed_interaction = self.embedding_interaction(interaction)
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)
        '''
        
        embed_cate_features = torch.cat(
                [self.cate_embedding_layers[cate_col](cate_input) for cate_input, cate_col \
                    in zip(cate_inputs, self.args.cate_cols)], 2)
    
        embed_num_features = torch.cat(
                [self.num_embedding_layers[num_col](num_input.unsqueeze(2)) for num_input, num_col \
                    in zip(num_inputs, self.args.num_cols)], 2)

        embed = torch.cat([embed_cate_features,
                           embed_num_features], 2)

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
        self.cate_embedding_layers, self.num_embedding_layers, self.comb_proj = init_layers(args)

        # # interaction은 현재 correct으로 구성되어있다. correct(1, 2) + padding(0)
        # self.embedding_interaction = nn.Embedding(3, self.hidden_dim//3)
        # self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim//3)
        # self.embedding_question = nn.Embedding(self.args.n_questions + 1, self.hidden_dim//3)
        # self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim//3)

        # # embedding combination projection
        # self.comb_proj = nn.Linear((self.hidden_dim//3)*4, self.hidden_dim)

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
        test, question, tag, _, mask, interaction, _ = input
        batch_size = interaction.size(0)

        # 신나는 embedding

        embed_interaction = self.embedding_interaction(interaction)
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)

        embed = torch.cat([embed_interaction,

                           embed_test,
                           embed_question,

                           embed_tag,], 2)

        X = self.comb_proj(embed)

        # Bert
        encoded_layers = self.encoder(inputs_embeds=X, attention_mask=mask)
        out = encoded_layers[0]
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out)
        preds = self.activation(out).view(batch_size, -1)

        return preds