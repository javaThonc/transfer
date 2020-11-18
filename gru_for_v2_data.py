import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import os
import copy
import json
import argparse
import datetime
import collections
import numpy as np
import pandas as pd

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

EPS = 1e-12
e = 2.718281828459


class DataLoader:

    def __init__(self, df_feature, df_label, device, batch_size=800, pin_memory=False):

        assert len(df_feature) == len(df_label)

        self.df_feature = df_feature.values
        self.df_label = df_label.values
        self.device =  device

        if pin_memory:
            self.df_feature = torch.tensor(self.df_feature, dtype=torch.float, device=self.device)
            self.df_label = torch.tensor(self.df_label, dtype=torch.float, device=self.device)

        self.index = df_label.index

        self.batch_size = batch_size
        self.pin_memory = pin_memory

        self.daily_count = df_label.groupby(level=0).size().values
        self.daily_index = np.roll(np.cumsum(self.daily_count), 1)
        self.daily_index[0] = 0

    @property
    def batch_length(self):

        if self.batch_size <= 0:
            return self.daily_length

        return len(self.df_label) // self.batch_size

    @property
    def daily_length(self):

        return len(self.daily_count)

    def iter_batch(self):

        if self.batch_size <= 0:
            yield from self.iter_daily()
            return

        indices = np.arange(len(self.df_label))
        np.random.shuffle(indices)

        for i in range(len(indices))[::self.batch_size]:
            if len(indices) - i < self.batch_size:
                break
            yield indices[i:i+self.batch_size] # NOTE: advanced indexing will cause copy

    def iter_daily(self):

        for idx, count in zip(self.daily_index, self.daily_count):
            yield slice(idx, idx + count) # NOTE: slice index will not cause copy

    def get(self, slc):

        outs = self.df_feature[slc], self.df_label[slc]

        if not self.pin_memory:
            outs = tuple(torch.tensor(x, dtype=torch.float, device=self.device) for x in outs)

        return outs + (self.index[slc],)


class GRU(nn.Module):

    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0):
        super().__init__()

        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc_out = nn.Linear(hidden_size, 1)

        self.d_feat = d_feat

    def forward(self, x):
        # x: [N, F*T]
        x = x.reshape(len(x), self.d_feat, -1) # [N, F, T]
        x = x.permute(0, 2, 1) # [N, T, F]
        out, _ = self.rnn(x)
        return self.fc_out(out[:, -1, :]).squeeze()


class MLP(nn.Module):

    def __init__(self, d_feat, hidden_size=256, num_layers=3, dropout=0.0):
        super().__init__()

        self.mlp = nn.Sequential()

        for i in range(num_layers):
            if i > 0:
                self.mlp.add_module('drop_%d'%i, nn.Dropout(dropout))
            self.mlp.add_module('fc_%d'%i, nn.Linear(
                d_feat if i == 0 else hidden_size, hidden_size))
            self.mlp.add_module('relu_%d'%i, nn.ReLU())

        self.mlp.add_module('fc_out', nn.Linear(hidden_size, 1))

    def forward(self, x):
        # feature
        # [N, F]
        return self.mlp(x).squeeze()

class LSTM(nn.Module):

    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0):
        super().__init__()

        self.rnn = nn.LSTM(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc_out = nn.Linear(hidden_size, 1)

        self.d_feat = d_feat

    def forward(self, x):
        # x: [N, F*T]
        x = x.reshape(len(x), self.d_feat, -1) # [N, F, T]
        x = x.permute(0, 2, 1) # [N, T, F]
        out, _ = self.rnn(x)
        return self.fc_out(out[:, -1, :]).squeeze()

class SFM(nn.Module):
    def __init__(self, d_feat=6, output_dim = 1, freq_dim = 10, hidden_size = 64, num_layers = 1,dropout_W = 0.0, dropout_U = 0.0):
        super().__init__()
        self.input_dim  = d_feat
        self.output_dim = output_dim
        self.freq_dim = freq_dim
        self.hidden_dim = hidden_size

        print(self.hidden_dim)
        print(hidden_size)
        self.W_i = nn.Parameter(init.xavier_uniform_(torch.empty((self.input_dim, self.hidden_dim))))
        self.U_i = nn.Parameter(init.orthogonal_(torch.empty(self.hidden_dim, self.hidden_dim)))
        self.b_i = nn.Parameter(torch.zeros(self.hidden_dim))

        self.W_ste = nn.Parameter(init.xavier_uniform_(torch.empty(self.input_dim, self.hidden_dim)))
        self.U_ste = nn.Parameter(init.orthogonal_(torch.empty(self.hidden_dim, self.hidden_dim)))
        self.b_ste = nn.Parameter(torch.ones(self.hidden_dim))

        self.W_fre = nn.Parameter(init.xavier_uniform_(torch.empty(self.input_dim, self.freq_dim)))
        self.U_fre = nn.Parameter(init.orthogonal_(torch.empty(self.hidden_dim, self.freq_dim)))
        self.b_fre = nn.Parameter(torch.ones(self.freq_dim))

        self.W_c = nn.Parameter(init.xavier_uniform_(torch.empty(self.input_dim, self.hidden_dim)))
        self.U_c = nn.Parameter(init.orthogonal_(torch.empty(self.hidden_dim, self.hidden_dim)))
        self.b_c = nn.Parameter(torch.zeros(self.hidden_dim))

        self.W_o = nn.Parameter(init.xavier_uniform_(torch.empty(self.input_dim, self.hidden_dim)))
        self.U_o = nn.Parameter(init.orthogonal_(torch.empty(self.hidden_dim, self.hidden_dim)))
        self.b_o = nn.Parameter(torch.zeros(self.hidden_dim))

        self.U_a = nn.Parameter(init.orthogonal_(torch.empty(self.freq_dim, 1)))
        self.b_a = nn.Parameter(torch.zeros(self.hidden_dim))

        self.W_p = nn.Parameter(init.xavier_uniform_(torch.empty(self.hidden_dim, self.output_dim)))
        self.b_p = nn.Parameter(torch.zeros(self.output_dim))
        
        self.activation = nn.Tanh()
        self.inner_activation = nn.Hardsigmoid()
        self.dropout_W, self.dropout_U = (dropout_W, dropout_U)
        self.states = []

    def init_states(self, x):
        init_state_h = torch.zeros_like(x)
        init_state_h = torch.sum(init_state_h, axis=1)
        reducer_s = torch.zeros((self.input_dim, self.hidden_dim))
        reducer_f = torch.zeros((self.hidden_dim, self.freq_dim))
        reducer_p = torch.zeros((self.hidden_dim, self.output_dim))
        init_state_h = torch.matmul(init_state_h, reducer_s)

        init_state_p = torch.matmul(init_state_h, reducer_p)
        
        init_state = torch.zeros_like(init_state_h)
        init_freq = torch.matmul(init_state_h, reducer_f)

        init_state = torch.reshape(init_state, (-1, self.hidden_dim, 1))
        init_freq = torch.reshape(init_freq, (-1, 1, self.freq_dim))
        
        init_state_S_re = init_state * init_freq
        init_state_S_im = init_state * init_freq
        
        init_state_time = torch.tensor(0.)

        self.states = [init_state_p, init_state_h, init_state_S_re, init_state_S_im, init_state_time, None, None, None]
    
    def forward(self, x):
        if(len(self.states)==0): #hasn't initialized yet
            self.init_states(x)
        get_constants(x)
        p_tm1 = states[0]
        h_tm1 = states[1]
        S_re_tm1 = states[2]
        S_im_tm1 = states[3]
        time_tm1 = states[4]
        B_U = states[5]
        B_W = states[6]
        frequency = states[7]

        x_i = torch.matmul(x * B_W[0], self.W_i) + self.b_i
        x_ste = torch.matmul(x * B_W[0], self.W_ste) + self.b_ste
        x_fre = torch.matmul(x * B_W[0], self.W_fre) + self.b_fre
        x_c = torch.matmul(x * B_W[0], self.W_c) + self.b_c
        x_o = torch.matmul(x * B_W[0], self.W_o) + self.b_o
        
        i = self.inner_activation(x_i + torch.matmul(h_tm1 * B_U[0], self.U_i))
        
        ste = self.inner_activation(x_ste + torch.matmul(h_tm1 * B_U[0], self.U_ste))
        fre = self.inner_activation(x_fre + torch.matmul(h_tm1 * B_U[0], self.U_fre))

        ste = torch.reshape(ste, (-1, self.hidden_dim, 1))
        fre = torch.reshape(fre, (-1, 1, self.freq_dim))
        f = ste * fre
        
        c = i * self.activation(x_c + torch.matmul(h_tm1 * B_U[0], self.U_c))

        time = time_tm1 + 1

        omega = torch.tensor(2*np.pi)* time * frequency
        re = torch.cos(omega)
        im = torch.sin(omega)
        
        c = torch.reshape(c, (-1, self.hidden_dim, 1))
        
        S_re = f * S_re_tm1 + c * re
        S_im = f * S_im_tm1 + c * im
        
        A = torch.square(S_re) + torch.square(S_im)

        A = torch.reshape(A, (-1, self.freq_dim))
        A_a = torch.matmul(A * B_U[0], self.U_a)
        A_a = torch.reshape(A_a, (-1, self.hidden_dim))
        a = self.activation(A_a + self.b_a)
        
        o = self.inner_activation(x_o + torch.matmul(h_tm1 * B_U[0], self.U_o))

        h = o * a
        p = torch.matmul(h, self.W_p) + self.b_p


        self.states = [p, h, S_re, S_im, time, None, None, None]

        return p

    def get_constants(self, x):
        constants.append([torch.tensor(1.) for _ in range(6)])
        constants.append([torch.tensor(1.) for _ in range(7)])
        array = np.array([float(ii)/self.freq_dim for ii in range(self.freq_dim)])
        constants.append([torch.tensor(array)])

        self.states[5:] = constants

def get_model(model_name):

    if model_name.upper() == 'LSTM':
        return LSTM

    if model_name.upper() == 'GRU':
        return GRU

    if model_name.upper() == 'MLP':
        return MLP

    if model_name.upper() == 'SFM':
        return SFM

    raise ValueError('unknown model name `%s`'%model_name)

def cal_ic(pred, label, args):
    return torch.mean(pred * label)

def metric_fn(pred, label, args):

    mask = torch.isfinite(label)
    if args.metric == 'IC':
        return cal_ic(pred[mask], label[mask], args)

    if args.metric == '' or args.metric == 'loss': # use loss
        return -loss_fn(pred[mask], label[mask], args)

    raise ValueError('unknown metric `%s`'%args.metric)


def mse(pred, label):
    loss = (pred - label)**2
    return torch.mean(loss)

def mae(pred, label):
    loss = (pred - label).abs()
    return torch.mean(loss)

def labelmse(pred, label):
    loss = torch.pow(label.abs(), args.exp_coef)*(pred - label)**2
    return torch.mean(loss)

def longmse(pred, label):
    loss = torch.pow(e, args.exp_coef*label)*(pred - label)**2
    return torch.mean(loss)

def labellogcosh(pred, label):
    loss = torch.pow(label.abs(), args.exp_coef)*torch.log(torch.cosh(pred - label))
    return torch.mean(loss)

def longlogcosh(pred, label):
    loss = torch.pow(e, args.exp_coef*label)*torch.log(torch.cosh(pred - label))
    return torch.mean(loss)

def loss_fn(pred, label, args):
    mask = ~torch.isnan(label)

    if args.loss == 'mse':
        return mse(pred[mask], label[mask])

    if args.loss == 'mae':
        return mae(pred[mask], label[mask])

    if args.loss == 'logcosh':
        return logcosh(pred[mask], label[mask])

    if args.loss == 'labelmse':
        return labelmse(pred[mask], label[mask])

    if args.loss == 'longmse':
        return longmse(pred[mask], label[mask])

    if args.loss == 'labellogcosh':
        return labellogcosh(pred[mask], label[mask])

    if args.loss == 'longlogcosh':
        return labellogcosh(pred[mask], label[mask])

    raise ValueError('unknown loss `%s`'%args.loss)


global_log_file = None
def pprint(*args):
    # print with UTC+8 time
    time = '['+str(datetime.datetime.utcnow()+
                   datetime.timedelta(hours=8))[:19]+'] -'
    print(time, *args, flush=True)

    if global_log_file is None:
        return
    with open(global_log_file, 'a') as f:
        print(time, *args, flush=True, file=f)


global_step = -1
def train_epoch(epoch, model, optimizer, train_loader, writer, args):

    global global_step

    model.train()

    for slc in tqdm(train_loader.iter_batch(), total=train_loader.batch_length):

        global_step += 1

        feature, label, _ = train_loader.get(slc)

        pred = model(feature)
        loss = loss_fn(pred, label, args)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 3.)
        optimizer.step()

        writer.add_scalar('Train/RunningLoss', loss.item(), global_step)


def test_epoch(epoch, model, test_loader, writer, args, prefix='Test'):

    model.eval()

    scores = []
    losses = []

    for slc in tqdm(test_loader.iter_daily(), desc=prefix, total=test_loader.daily_length):

        feature, label, _ = test_loader.get(slc)
        with torch.no_grad():
            pred = model(feature)

        loss = loss_fn(pred, label, args)
        losses.append(loss.item())

        score = metric_fn(pred, label, args)
        scores.append(score.item())

    writer.add_scalar(prefix+'/Loss', np.mean(losses), epoch)
    writer.add_scalar(prefix+'/std(Loss)', np.std(losses), epoch)
    writer.add_scalar(prefix+'/'+args.metric, np.mean(scores), epoch)
    writer.add_scalar(prefix+'/std('+args.metric+')', np.std(scores), epoch)

    return np.mean(losses), np.mean(scores)


def inference(model, data_loader):

    model.eval()

    preds = []
    for slc in tqdm(data_loader.iter_daily(), total=data_loader.daily_length):

        feature, label, index = data_loader.get(slc)
        with torch.no_grad():
            pred = model(feature)

        preds.append(pd.DataFrame({
            'score': pred.cpu().numpy(),
            'label': label.cpu().numpy(),
            }, index=index))

    preds = pd.concat(preds, axis=0)

    return preds


def create_loaders(args, device):

    df = pd.read_pickle('./' + args.dset+'.pkl')

    # NOTE: we always assume the last column is label
    df_feature = df.iloc[:, 0:360]
    df_label = df[args.label]*100
    df_label[df_label > np.percentile(df_label, 99)] = np.percentile(df_label, 99)
    df_label[df_label < np.percentile(df_label, 1)] = np.percentile(df_label, 1)

    slc = slice(pd.Timestamp(args.train_start_date), pd.Timestamp(args.train_end_date))
    train_loader = DataLoader(df_feature.loc[slc], df_label.loc[slc], device=device,
                              batch_size=args.batch_size, pin_memory=args.pin_memory)

    slc = slice(pd.Timestamp(args.valid_start_date), pd.Timestamp(args.valid_end_date))
    valid_loader = DataLoader(df_feature.loc[slc], df_label.loc[slc], device=device, pin_memory=False)

    slc = slice(pd.Timestamp(args.test_start_date), pd.Timestamp(args.test_end_date))
    test_loader = DataLoader(df_feature.loc[slc], df_label.loc[slc], device=device, pin_memory=False)

    return train_loader, valid_loader, test_loader


def main(args):

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    suffix = "%s_dh%s_dn%s_drop%s_lr%s_bs%s_seed%s%s_label%s_dset%s"%(
        args.model_name, args.hidden_size, args.num_layers, args.dropout,
        args.lr, args.batch_size, args.seed, args.annot, args.label, args.dset
    )
    if args.loss != 'logcosh':
        suffix += '_loss%s'%(args.loss)

    if args.exp_coef :
        suffix += '_expcoef%d'%(args.exp_coef)

    output_path = args.outdir
    if not output_path:
        output_path = './output/' + suffix
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if not args.overwrite and os.path.exists(output_path+'/'+'info.json'):
        print('already runned, exit.')
        return

    writer = SummaryWriter(log_dir=output_path)

    global global_log_file
    global_log_file = output_path + '/' + 'run.log'

    pprint('create model...')
    device = 'cuda:%d'%(args.cuda) if torch.cuda.is_available() else 'cpu'
    model = get_model(args.model_name)(d_feat = args.d_feat, hidden_size = args.hidden_size, num_layers = args.num_layers, dropout_W = args.dropout, dropout_U = args.dropout)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    pprint('create loaders...')
    train_loader, valid_loader, test_loader = create_loaders(args=args, device=device)

    best_score = -np.inf
    best_epoch = 0
    stop_round = 0
    best_param = copy.deepcopy(model.state_dict())
    params_list = collections.deque(maxlen=args.smooth_steps)
    for epoch in range(args.n_epochs):
        pprint('Epoch:', epoch)

        pprint('training...')
        train_epoch(epoch, model, optimizer, train_loader, writer, args)
        torch.save(model.state_dict(), output_path+'/model.bin.e'+str(epoch))
        torch.save(optimizer.state_dict(), output_path+'/optimizer.bin.e'+str(epoch))

        pprint('evaluating...')
        train_loss, train_score = test_epoch(epoch, model, train_loader, writer, args, prefix='Train')
        val_loss, val_score = test_epoch(epoch, model, valid_loader, writer, args, prefix='Valid')
        test_loss, test_score = test_epoch(epoch, model, test_loader, writer, args, prefix='Test')

        pprint('train %.6f, valid %.6f, test %.6f'%(train_score, val_score, test_score))

        if val_score > best_score:
            best_score = val_score
            stop_round = 0
            best_epoch = epoch
            best_param = copy.deepcopy(model.state_dict())
        else:
            stop_round += 1
            if stop_round >= args.early_stop:
                pprint('early stop')
                break

    pprint('best score:', best_score, '@', best_epoch)
    model.load_state_dict(best_param)
    torch.save(best_param, output_path+'/model.bin')

    best_param = torch.load(output_path + '/model.bin')
    model.load_state_dict(best_param)

    pprint('inference...')
    res = dict()
    for name in ['train', 'valid', 'test']:

        pred = inference(model, eval(name+'_loader'))
        pred.to_pickle(output_path+'/pred.pkl.'+name)

        rank_ic = pred.groupby(level='datetime').apply(
            lambda x: x.label.corr(x.score, method='spearman'))

        pprint(('%s: Rank IC %.3f, Rank ICIR %.3f')%(
            name, rank_ic.mean(), rank_ic.mean()/rank_ic.std()))

        res[name+'-RankIC'] = rank_ic.mean()
        res[name+'-RankICIR'] = rank_ic.mean() / rank_ic.std()

        if name == 'test':
            pred.iloc[:,0].to_pickle(output_path+'/pred_score.pkl')

    pprint('finished.')


def parse_args():

    parser = argparse.ArgumentParser()

    # model
    parser.add_argument('--model_name', default='GRU')
    parser.add_argument('--d_feat', type=int, default=6)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.0)

    # training
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--early_stop', type=int, default=20)
    parser.add_argument('--smooth_steps', type=int, default=5)
    parser.add_argument('--metric', default='') # '' refers to loss
    parser.add_argument('--loss', default='mse')
    parser.add_argument('--exp_coef', type=int, default=0)

    # data
    parser.add_argument('--pin_memory', action='store_false', default=True)
    parser.add_argument('--batch_size', type=int, default=800) # -1 indicate daily batch
    parser.add_argument('--dset', default='sp500_data_tanh')
    parser.add_argument('--label', default='LABEL0') # specify other labels
    parser.add_argument('--train_start_date', default='2008-01-01')
    parser.add_argument('--train_end_date', default='2014-12-31')
    parser.add_argument('--valid_start_date', default='2015-01-01')
    parser.add_argument('--valid_end_date', default='2016-12-31')
    parser.add_argument('--test_start_date', default='2017-01-01')
    parser.add_argument('--test_end_date', default='2020-08-01')

    # other
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--annot', default='')
    parser.add_argument('--outdir', default='')
    parser.add_argument('--overwrite', action='store_true', default=False)
    parser.add_argument('--cuda', type=int, default=1)

    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = parse_args()
    main(args)
