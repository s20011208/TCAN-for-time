import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from process.setting import OPT
import math
from model.mhal import MultiHeadAttention

opt = OPT()

class Embedding(nn.Module):
    def __init__(self, N, embded_dim):
        super(Embedding, self).__init__()
        self.embed = nn.Embedding(N, embded_dim)

    def forward(self, x):
        return self.embed(x)

class Time2vec(nn.Module):
    def __init__(self, embed_dim):
        super(Time2vec, self).__init__()
        self.W = nn.Parameter(torch.empty(1, embed_dim))
        self.P = nn.Parameter(torch.empty(1, embed_dim))
        self.w = nn.Parameter(torch.empty(1,1))
        self.p = nn.Parameter(torch.empty(1,1))
        self.alpha = nn.Parameter(torch.empty(1,1))
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.P, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.p, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.alpha, a=math.sqrt(5))

    def forward(self, x, ts):
        original = self.w * ts + self.p
        power_low = self.alpha * torch.sqrt(ts)
        harmonic = torch.cos(torch.matmul(ts, self.W) + self.P)
        time_embedding = torch.cat([original, power_low, harmonic], dim=-1)
        return torch.cat([x, time_embedding], dim=-1) # time embedding

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super(FeedForward, self).__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.dropout(F.relu(self.w1(x))))

class CGAT(nn.Module):
    def __init__(self, d_model):
        super(CGAT, self).__init__()
        self.mha = MultiHeadAttention(model_dim=d_model, num_heads=opt.nhead, dropout=opt.dropout)

    def forward(self, x, attn_mask):
        o, _ = self.mha(x, x, x, attn_mask=attn_mask)
        return o

class CSLA(nn.Module):
    def __init__(self, d_model):
        super(CSLA, self).__init__()
        self.num_layers = opt.num_layers
        self.directional = 2
        self.hidden_dim = opt.hidden_dim

        self.rnn = nn.LSTM(input_size=d_model, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                           bidirectional=True, batch_first=True)
        self.mha = MultiHeadAttention(model_dim=d_model, num_heads=opt.nhead, dropout=opt.dropout)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, x, real_len, attn_mask):
        # two layers LSTM
        input = rnn_utils.pack_padded_sequence(x, real_len, batch_first=True)
        state = (torch.zeros(self.num_layers * self.directional, x.size(0), self.hidden_dim),
                 torch.zeros(self.num_layers * self.directional, x.size(0), self.hidden_dim))
        output, hn = self.rnn(input, state)  # output:(batch, seqlen, num_directions * hidden_size), hn:(batch,num_layers * num_directions, hidden_size)
        output, _ = rnn_utils.pad_packed_sequence(output, batch_first=True)
        output = self.layernorm(output)
        csla_out, _ = self.mha(output, output, output, attn_mask=attn_mask)
        return csla_out

def get_atten_mask(src):
    batch_size = src.size(0)
    seq_len = src.size(1)
    pad_attn_mask = src.data.eq(0).unsqueeze(1)
    return pad_attn_mask.expand(batch_size, seq_len, seq_len)

def get_adj_mask(adj):
    # -1e9
    pad_attn_mask = adj.data.eq(0)
    return pad_attn_mask

class TCAN(pl.LightningModule):
    def __init__(self, N, in_feats, out_feats):
        super(TCAN, self).__init__()
        self.hidden_dim = opt.hidden_dim

        self.embedding = Embedding(N, opt.embed_dim)
        self.time_embdding = Time2vec(opt.time_embed_dim)
        self.layernorm = nn.LayerNorm(in_feats)

        self.cgat1 = CGAT(in_feats)
        self.cgat2 = CGAT(in_feats)
        self.cgat3 = CGAT(in_feats)

        self.csla = CSLA(in_feats)

        self.fc1 = nn.Linear(self.hidden_dim * 4, self.hidden_dim * 2)
        self.fc2 = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2)
        self.fc3 = nn.Linear(self.hidden_dim * 2, out_feats)

        self.save_hyperparameters()

    def forward(self, batch):
        src, real_len, ts, adj = batch
        # embedding
        ts = torch.unsqueeze(ts, -1)
        X = self.time_embdding(self.embedding(src), ts)
        X = self.layernorm(X)

        # two layers gat
        adj_mask = get_adj_mask(adj)
        cgo = self.cgat1(X, adj_mask)
        cgo = self.cgat2(cgo, adj_mask)
        cgo = self.cgat3(cgo, adj_mask)
        cgat_output = torch.sum(cgo, 1)

        # two layers LSTM
        atten_mask = get_atten_mask(src)
        co = self.csla(X, real_len, atten_mask)
        csla_output = torch.sum(co, 1)
        # concat
        concat_out = torch.cat([cgat_output, csla_output], dim=1)
        concat_out = F.dropout(concat_out, training=self.training, p=opt.dropout)

        # three layers MLP
        o = F.dropout(self.fc1(concat_out), training=self.training, p=opt.dropout)
        o = F.relu(o)
        o = F.dropout(self.fc2(o), training=self.training, p=opt.dropout)
        o = F.relu(o)
        o = self.fc3(o)

        return o

    def configure_optimizers(self):
        opimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.weight_dacy)
        return opimizer

    def training_step(self, train_batch, batch_idx):
        x, ts, adj, y, real_len = train_batch
        pre = self([x, real_len, ts, adj])
        loss = F.mse_loss(pre, y.view(-1, 1))
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, ts, adj, y, real_len = val_batch
        real_len = real_len.cpu().to(torch.int64)
        pre = self([x, real_len, ts, adj])
        loss = F.mse_loss(pre, y.view(-1, 1))
        self.log('val_loss', loss)

    def test_step(self, test_batch, batch_idx):
        x, ts, adj, y, real_len = test_batch
        real_len = real_len.cpu().to(torch.int64)
        y_hat = self([x, real_len, ts, adj])
        loss = F.mse_loss(y_hat, y.view(-1, 1))
        self.log('test_loss', loss)




