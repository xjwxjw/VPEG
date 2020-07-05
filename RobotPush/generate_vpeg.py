import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import os
import random
from torch.autograd import Variable
from torch.utils.data import DataLoader
import utils
import itertools
import progressbar
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
parser.add_argument('--batch_size', default=16, type=int, help='batch size')
parser.add_argument('--log_dir', default='logs', help='base directory to save logs')
parser.add_argument('--model_dir', default='', help='base directory to save logs')
parser.add_argument('--name', default='', help='identifier for directory')
parser.add_argument('--data_root', default='data', help='root directory for data')
parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
parser.add_argument('--niter', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--epoch_size', type=int, default=600, help='epoch size')
parser.add_argument('--image_width', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--channels', default=3, type=int)
parser.add_argument('--dataset', default='kth', help='dataset to train with')
parser.add_argument('--n_past', type=int, default=5, help='number of frames to condition on')
parser.add_argument('--n_future', type=int, default=10, help='number of frames to predict during training')
parser.add_argument('--n_eval', type=int, default=20, help='number of frames to predict during eval')
parser.add_argument('--rnn_size', type=int, default=1024, help='dimensionality of hidden layer')
parser.add_argument('--prior_rnn_layers', type=int, default=1, help='number of layers')
parser.add_argument('--posterior_rnn_layers', type=int, default=1, help='number of layers')
parser.add_argument('--predictor_rnn_layers', type=int, default=2, help='number of layers')
parser.add_argument('--z_dim', type=int, default=24, help='dimensionality of z_t')
parser.add_argument('--g_dim', type=int, default=128, help='dimensionality of encoder output vector and decoder input vector')
parser.add_argument('--beta', type=float, default=0.001, help='weighting on KL to prior')
parser.add_argument('--model', default='vgg', help='model type (dcgan | vgg)')
parser.add_argument('--data_threads', type=int, default=5, help='number of data loading threads')
parser.add_argument('--num_digits', type=int, default=2, help='number of digits for moving mnist')
parser.add_argument('--last_frame_skip', action='store_true', help='if true, skip connections go between frame t and frame t+t rather than last ground truth frame')



opt = parser.parse_args()
if opt.model_dir != '':
    # load model and continue training from checkpoint
    saved_model = torch.load('%s/model.pth' % opt.model_dir)
    optimizer = opt.optimizer
    model_dir = opt.model_dir
    opt = saved_model['opt']
    opt.optimizer = optimizer
    opt.model_dir = model_dir
    opt.log_dir = '%s/continued' % opt.log_dir
else:
    name = 'model=%s%dx%d-rnn_size=%d-predictor-posterior-prior-rnn_layers=%d-%d-%d-n_past=%d-n_future=%d-lr=%.4f-g_dim=%d-z_dim=%d-last_frame_skip=%s-beta=%.7f%s' % (opt.model, opt.image_width, opt.image_width, opt.rnn_size, opt.predictor_rnn_layers, opt.posterior_rnn_layers, opt.prior_rnn_layers, opt.n_past, opt.n_future, opt.lr, opt.g_dim, opt.z_dim, opt.last_frame_skip, opt.beta, opt.name)
    if opt.dataset == 'smmnist':
        opt.log_dir = '%s/%s-%d/%s' % (opt.log_dir, opt.dataset, opt.num_digits, name)
    else:
        opt.log_dir = '%s/%s/%s' % (opt.log_dir, opt.dataset, name)

os.makedirs('%s/gen/' % opt.log_dir, exist_ok=True)
os.makedirs('%s/plots/' % opt.log_dir, exist_ok=True)

print("Random Seed: ", opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
dtype = torch.cuda.FloatTensor


# ---------------- load the models  ----------------

print(opt)

# ---------------- optimizers ----------------
if opt.optimizer == 'adam':
    opt.optimizer = optim.Adam
elif opt.optimizer == 'rmsprop':
    opt.optimizer = optim.RMSprop
elif opt.optimizer == 'sgd':
    opt.optimizer = optim.SGD
else:
    raise ValueError('Unknown optimizer: %s' % opt.optimizer)


import models.lstm as lstm_models
if opt.model == 'dcgan':
    if opt.image_width == 64:
        import models.dcgan_64 as model 
    elif opt.image_width == 128:
        import models.dcgan_128 as model  
elif opt.model == 'vgg':
    if opt.image_width == 64:
        import models.vgg_64 as model
    elif opt.image_width == 128:
        import models.vgg_128 as model
else:
    raise ValueError('Unknown model: %s' % opt.model)
       
saved_model = torch.load('/media/xjw/E4C04590C04569BE/svg/logs/bair-match/bair-match/model=vgg64x64-rnn_size=512-predictor-posterior-prior-rnn_layers=3-1-1-n_past=5-n_future=10-lr=0.0010-g_dim=128-z_dim=64-last_frame_skip=False-beta=0.0001000/model.pth')
decoder = saved_model['decoder']
encoder = saved_model['encoder']
frame_predictor = saved_model['frame_predictor']
posterior = saved_model['posterior']

# --------- loss functions ------------------------------------
mse_criterion = nn.MSELoss()
def kl_criterion(mu1, logvar1, mu2, logvar2):
    # KL( N(mu_1, sigma2_1) || N(mu_2, sigma2_2)) = 
    #   log( sqrt(
    # 
    sigma1 = logvar1.mul(0.5).exp() 
    sigma2 = logvar2.mul(0.5).exp() 
    kld = torch.log(sigma2/sigma1) + (torch.exp(logvar1) + (mu1 - mu2)**2)/(2*torch.exp(logvar2)) - 1/2
    return kld.sum() / opt.batch_size

# --------- transfer to gpu ------------------------------------
frame_predictor.cuda()
posterior.cuda()
# prior.cuda()
encoder.cuda()
decoder.cuda()

# --------- load a dataset ------------------------------------
train_data, test_data = utils.load_dataset(opt)

train_loader = DataLoader(train_data,
                          num_workers=opt.data_threads,
                          batch_size=opt.batch_size,
                          shuffle=True,
                          drop_last=True,
                          pin_memory=True)
test_loader = DataLoader(test_data,
                         num_workers=opt.data_threads,
                         batch_size=opt.batch_size,
                         shuffle=True,
                         drop_last=True,
                         pin_memory=True)

def get_training_batch():
    while True:
        for sequence in train_loader:
            sequence_0, sequence_match = sequence
            batch_0 = utils.normalize_data(opt, dtype, sequence_0)
            batch_match = []
            for i in range(5):
                batch_match.append(utils.normalize_data(opt, dtype, sequence_match[i]))
            yield batch_0, batch_match
training_batch_generator = get_training_batch()

def get_testing_batch():
    while True:
        for sequence in test_loader:
            sequence_0, sequence_match = sequence
            batch_0 = utils.normalize_data(opt, dtype, sequence_0)
            batch_match = []
            for i in range(5):
                batch_match.append(utils.normalize_data(opt, dtype, sequence_match[i]))
            yield batch_0, batch_match
testing_batch_generator = get_testing_batch()

def plot(x, y, epoch):
    nsample = 20 
    gen_seq = [[] for i in range(nsample)]
    gt_seq = [x[i] for i in range(len(x))]

    for s in range(nsample):
        frame_predictor.hidden = frame_predictor.init_hidden()
        posterior.hidden = posterior.init_hidden()
        gen_seq[s].append(x[0])
        h_match_prev = [encoder(y[t][0])[0].detach() for t in range(5)]
        for i in range(1, opt.n_eval):
            h_match = [encoder(y[t][i])[0].detach() for t in range(5)]
            h_target = encoder(x[i])
            if i < opt.n_past:
                h = encoder(x[i-1])	
                h, skip = h
            else:
                # h, _ = encoder(x_pred)	
                h = h_pred
            h_target, _ = h_target
            h = h.detach()
            h_target = h_target.detach()
            ref_feat = torch.cat([torch.mean(torch.cat([h_m.unsqueeze(1) for h_m in h_match], 1), 1), torch.std(torch.cat([h_m.unsqueeze(1) for h_m in h_match], 1), 1)], -1)
            z_t, _, _ = posterior(torch.cat([h_match[m] - h_match_prev[m] for m in range(5)], 1))
            h_pred = frame_predictor(h)
            x_pred = decoder([h_pred, skip])
            if i < opt.n_past:
                gen_seq[s].append(x[i])
            else:
                gen_seq[s].append(x_pred)
            h_match_prev = h_match
            # print(i, h_match[0][1,1].detach().cpu().numpy(),
            #     h_target[1,1].detach().cpu().numpy(),
            #     h_pred[1,1].detach().cpu().numpy())

    to_plot = []
    gifs = [ [] for t in range(opt.n_eval) ]
    nrow = min(opt.batch_size, 10)
    for i in range(nrow):
        # ground truth sequence
        row = [] 
        for t in range(opt.n_eval):
            row.append(gt_seq[t][i])
        to_plot.append(row)

        # best sequence
        min_mse = 1e7
        for s in range(nsample):
            mse = 0
            for t in range(opt.n_eval):
                mse +=  torch.sum( (gt_seq[t][i].data.cpu() - gen_seq[s][t][i].data.cpu())**2 )
            if mse < min_mse:
                min_mse = mse
                min_idx = s

        s_list = [min_idx, 
                  np.random.randint(nsample), 
                  np.random.randint(nsample), 
                  np.random.randint(nsample), 
                  np.random.randint(nsample)]
        for ss in range(len(s_list)):
            s = s_list[ss]
            row = []
            for t in range(opt.n_eval):
                row.append(gen_seq[s][t][i]) 
            to_plot.append(row)
        for t in range(opt.n_eval):
            row = []
            row.append(gt_seq[t][i])
            for ss in range(len(s_list)):
                s = s_list[ss]
                row.append(gen_seq[s][t][i])
            for ss in range(5):
                row.append(y[ss][t][i])
            gifs[t].append(row)

    fname = '%s/gen/sample_%d.gif' % (opt.log_dir, epoch) 
    utils.save_gif(fname, gifs)

# --------- testing loop ------------------------------------
for epoch in range(opt.niter):
    encoder.train()
    decoder.train()
    frame_predictor.eval()
    posterior.eval()

    with torch.no_grad():
        x, y = next(testing_batch_generator)
        plot(x, y, epoch)
        

