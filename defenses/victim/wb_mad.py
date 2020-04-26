import argparse
import os.path as osp
import os
import json
import pickle

import numpy as np

from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import grad

from knockoff.utils.type_checks import TypeCheck
import knockoff.utils.model as model_utils
import knockoff.models.zoo as zoo
from knockoff import datasets

from defenses.victim import Blackbox
from defenses.utils.projection import euclidean_proj_l1ball, euclidean_proj_simplex
from .bb_mad import MAD

import matplotlib.pyplot as plt

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"


class MAD_WB(Blackbox):
    def __init__(self, epsilon=None, optim='linesearch', model_adv_proxy=None, max_grad_layer=None, ydist='l1',
                 oracle='extreme', model_adv=None, model_adv_proxy_notrain=False, out_path=None, disable_jacobian=False,
                 attacker_argmax=False, adv_optimizer='sgd', objmax=False, log_prefix='', *args, **kwargs):
        super().__init__(*args, **kwargs)
        print('=> MAD ({})'.format([self.dataset_name, epsilon, optim, ydist, oracle]))

        self.epsilon = epsilon
        self.out_path = out_path
        self.disable_jacobian = bool(disable_jacobian)
        if self.disable_jacobian:
            print('')
            print('!!!WARNING!!! Using G = eye(K)')
            print('')

        self.attacker_argmax = bool(attacker_argmax)
        if self.attacker_argmax:
            print('')
            print('!!!WARNING!!! Argmax of perturbed probabilities used to train attacker model')
            print('')

        self.objmax = bool(objmax)

        '''
            Here, we refer to three models:
            a) self.model: Victim's model (already initialized by super)
            b) self.model_adv: Adversary's model
            c) self.model_adv_proxy: Proxy to adversary's model
        '''

        '''(b) self.model_adv: Adversary's model '''
        # Unlike BreakSGD, here we assume a perfect-knowledge adversary - which will also be trained online
        # Initialize the adv model
        print('=> Initializing adv model compatible with: ', self.model_arch, self.modelfamily, self.dataset_name,
              self.num_classes)
        if model_adv is not None and osp.isdir(model_adv):
            model_adv = osp.join(model_adv, 'checkpoint.pth.tar')
        assert model_adv is None or osp.exists(model_adv)
        self.model_adv = zoo.get_net(self.model_arch, self.modelfamily, pretrained=model_adv,
                                     num_classes=self.num_classes)
        self.adv_optimizer = adv_optimizer
        assert adv_optimizer in ['sgd', 'sgdm', 'adam']
        if self.adv_optimizer == 'sgd':
            self.model_adv_optimizer = torch.optim.SGD(self.model_adv.parameters(), lr=0.1 / 64)
        elif self.adv_optimizer == 'sgdm':
            self.model_adv_optimizer = torch.optim.SGD(self.model_adv.parameters(), lr=0.1 / 64, momentum=0.5)
        elif self.adv_optimizer == 'adam':
            self.model_adv_optimizer = torch.optim.Adam(self.model_adv.parameters(), lr=0.001 / 64)
        else:
            raise ValueError('Unrecognized optimizer')
        self.model_adv = self.model_adv.to(self.device)
        self.model_adv.train()

        '''(c) self.model_adv_proxy: Proxy to adversary's model'''
        self.model_adv_proxy_notrain = bool(model_adv_proxy_notrain)
        if model_adv_proxy is None:
            # Perfect Knowledge (White-box attacker)
            self.model_adv_proxy = self.model_adv
        else:
            if osp.isdir(model_adv_proxy):
                model_adv_proxy = osp.join(model_adv_proxy, 'checkpoint.pth.tar')
            assert osp.exists(model_adv_proxy), 'Does not exist: {}'.format(model_adv_proxy)
            print('=== Models used for experiment ===')
            print('F_V        : ', self.model_dir)
            print('F_A        : ', osp.dirname(model_adv))
            print('F_A (proxy): ', osp.dirname(model_adv_proxy))
            print('F_A (proxy) trained online?: ', not self.model_adv_proxy_notrain)
            print('==================================')
            # Gray-box attacker
            self.model_adv_proxy = zoo.get_net(self.model_arch, self.modelfamily, pretrained=model_adv_proxy,
                                               num_classes=self.num_classes)
            if self.adv_optimizer == 'sgd':
                self.model_adv_proxy_optimizer = torch.optim.SGD(self.model_adv.parameters(), lr=0.1 / 64)
            elif self.adv_optimizer == 'sgdm':
                self.model_adv_proxy_optimizer = torch.optim.SGD(self.model_adv.parameters(), lr=0.1 / 64, momentum=0.5)
            elif self.adv_optimizer == 'adam':
                self.model_adv_proxy_optimizer = torch.optim.Adam(self.model_adv.parameters())
            else:
                raise ValueError('Unrecognized optimizer')
            self.model_adv_proxy = self.model_adv_proxy.to(self.device)

        # To compute stats
        self.dataset = datasets.__dict__[self.dataset_name]
        self.modelfamily = datasets.dataset_to_modelfamily[self.dataset_name]
        self.train_transform = datasets.modelfamily_to_transforms[self.modelfamily]['train']
        self.test_transform = datasets.modelfamily_to_transforms[self.modelfamily]['test']
        self.trainset = self.dataset(train=True, transform=self.train_transform)
        self.testset = self.dataset(train=False, transform=self.test_transform)
        self.test_loader = DataLoader(self.testset, batch_size=128, shuffle=False, num_workers=5)
        self.best_test_acc = 0.
        # Also keep a mini-testset to eval victim model using current strategy
        self.minitestset = self.dataset(train=False, transform=self.test_transform)
        self.minitestset = torch.utils.data.Subset(self.minitestset, indices=np.arange(1000))
        self.minitest_loader = DataLoader(self.minitestset, batch_size=1, shuffle=False, num_workers=1)

        self.K = len(self.testset.classes)
        self.D = None

        self.ydist = ydist
        assert ydist in ['l1', 'l2', 'kl']

        # Which oracle to use
        self.oracle = oracle
        assert self.oracle in ['extreme', 'random', 'argmin', 'argmax']

        # Which algorithm to use to optimize
        self.optim = optim
        assert optim in ['linesearch', 'projections', 'greedy']

        # Gradients from which layer to use?
        assert max_grad_layer in [None, 'all']
        self.max_grad_layer = max_grad_layer

        # Track some data for debugging
        self.queries = []  # List of (x_i, y_i, y_i_prime, distance)
        self.run_id = str(datetime.now())
        self.log_path = osp.join(out_path, 'online.log.tsv')
        if not osp.exists(self.log_path):
            with open(self.log_path, 'w') as wf:
                columns = ['run_id', 'epoch', 'split', 'loss', 'accuracy', 'best_accuracy', 'l1_mean', 'l1_std',
                           'l2_mean', 'l2_std', 'kl_mean', 'kl_std']
                wf.write('\t'.join(columns) + '\n')

    def calc_delta(self, x, y, debug=False):
        # Jacobians G
        if self.disable_jacobian or self.oracle in ['random', 'argmin']:
            G = torch.eye(self.K).to(self.device)
        else:
            G = MAD.compute_jacobian_nll(x, self.model_adv_proxy, device=self.device, K=self.K)
        if self.D is None:
            self.D = G.shape[1]

        # y* via oracle
        if self.oracle == 'random':
            ystar, ystar_val = MAD.oracle_rand(G, y)
        elif self.oracle == 'extreme':
            ystar, ystar_val = MAD.oracle_extreme(G, y, max_over_obj=self.objmax)
        elif self.oracle == 'argmin':
            ystar, ystar_val = MAD.oracle_argmin(G, y)
        elif self.oracle == 'argmax':
            ystar, ystar_val = MAD.oracle_argmax_preserving(G, y, max_over_obj=self.objmax)
        else:
            raise ValueError()

        # y* maybe outside the feasible set - project it back
        if self.optim == 'linesearch':
            delta = MAD.linesearch(G, y, ystar, self.ydist, self.epsilon)
        elif self.optim == 'projections':
            delta = MAD.projections(G, y, ystar, self.ydist, self.epsilon)
        elif self.optim == 'greedy':
            raise NotImplementedError()
        else:
            raise ValueError()

        # Calc. final objective values
        ytilde = y + delta
        objval = MAD.calc_objective(ytilde, y, G)
        objval_surrogate = MAD.calc_surrogate_objective(ytilde, y, G)

        return delta, objval, objval_surrogate

    def __call__(self, x):
        TypeCheck.multiple_image_blackbox_input_tensor(x)  # of shape B x C x H x W
        assert x.shape[0] == 1, 'Currently only supports B=1'

        with torch.no_grad():
            x = x.to(self.device)
            z_v = self.model(x)  # Victim's predicted logits
            y_v = F.softmax(z_v, dim=1).detach()
            self.call_count += x.shape[0]

        y_prime = []

        # No batch support yet. So, perturb individually.
        x_i = x[0].unsqueeze(0)
        y_v_i = y_v[0]

        with torch.enable_grad():
            delta_i, objval, sobjval = self.calc_delta(x_i, y_v_i)

        y_prime_i = y_v_i + delta_i

        # ---------------------- Sanity checks
        # ---------- 1. No NaNs
        assert torch.isnan(delta_i).sum().item() == 0., ' y = {}\n delta = {}'.format(y_v_i, delta_i)
        # ---------- 2. Constraints are met
        if not MAD.is_in_simplex(y_prime_i):
            print('[WARNING] Simplex contraint failed (i = {})'.format(self.call_count))
        if not MAD.is_in_dist_ball(y_v_i, y_prime_i, self.ydist, self.epsilon):
            _dist = MAD.calc_distance(y_v_i, y_prime_i, self.ydist)
            print('[WARNING] Distance contraint failed (i = {}, dist = {:.4f} > {:.4f})'.format(self.call_count,
                                                                                                _dist,
                                                                                                self.epsilon))

        self.queries.append((y_v_i.cpu().detach().numpy(), y_prime_i.cpu().detach().numpy(),
                             objval.cpu().detach().numpy(), sobjval.cpu().detach().numpy()))

        y_prime = y_prime_i
        if len(y_prime.shape) != 2:
            y_prime.unsqueeze_(0)

        if self.attacker_argmax:
            y_prime = self.truncate_output(y_prime, topk=1, rounding=0)

        # Perform training step of adv's model using this example
        self.adv_train_step(x, y_prime)

        if (not self.model_adv_proxy_notrain) and (self.model_adv_proxy != self.model_adv):
            self.proxy_train_step(x, y_prime)

        if self.call_count % 1000 == 0:
            self.log_whitebox()

        return y_prime

    def adv_train_step(self, x, y):
        # Perform training step of adv's model using this example
        with torch.enable_grad():
            z_a = self.model_adv(x)
            loss = model_utils.soft_cross_entropy(z_a, y)
            self.model_adv_optimizer.zero_grad()
            loss.backward()
            self.model_adv_optimizer.step()

    def proxy_train_step(self, x, y):
        with torch.enable_grad():
            # Also make a simultaneous update on the proxy model
            z_a = self.model_adv_proxy(x)
            loss = model_utils.soft_cross_entropy(z_a, y)
            self.model_adv_proxy_optimizer.zero_grad()
            loss.backward()
            self.model_adv_proxy_optimizer.step()

    def log_whitebox(self):
        epoch = self.call_count / len(self.trainset)
        # Evaluate knockoff model
        test_loss, test_acc = model_utils.test_step(self.model_adv, self.test_loader, nn.CrossEntropyLoss(),
                                                    self.device, epoch=epoch, silent=True)
        self.best_test_acc = max(test_acc, self.best_test_acc)
        l1_mean, l1_std, l2_mean, l2_std, kl_mean, kl_std = MAD.calc_query_distances(self.queries)
        print('[{}] [ADV] Loss = {:.4f}\tTest-acc = {:.2f}\t'
              'L1 = {:.2f}\tL2 = {:.2f}\tKL = {:.2f}'.format(self.call_count, test_loss, test_acc, l1_mean, l2_mean,
                                                             kl_mean))

        # Evaluate victim model
        mcorrect, mtotal, mloss = 0, 0, 0.
        ml1s, ml2s, mkls = [], [], []  # Track distances between (y, y')
        for mx, my in self.minitest_loader:
            mx = mx.to(self.device)
            mz_v = self.model(mx)
            my_v = F.softmax(mz_v, dim=1)
            with torch.enable_grad():
                if self.epsilon > 0.:
                    mdelta, *_ = self.calc_delta(mx, my_v[0])
                else:
                    mdelta = torch.zeros_like(my_v)
            assert torch.isnan(mdelta).sum().item() == 0., ' y = {}\n delta = {}'.format(my_v, mdelta)
            my_prime = my_v.detach().cpu() + mdelta.cpu()
            my_prime += 1e-7
            my_prime /= my_prime.sum()
            _, mpredicted = my_prime.max(1)
            mloss += F.nll_loss(torch.log(my_prime), my).item()

            # Track distances
            ml1s.append((my_v.cpu() - my_prime).norm(p=1).item())
            ml2s.append((my_v.cpu() - my_prime).norm(p=2).item())
            mkls.append(F.kl_div(my_prime.log(), my_v.cpu(), reduction='sum').item())

            mtotal += my.size(0)
            mcorrect += mpredicted.eq(my).sum().item()
        vtest_loss, vtest_acc = mloss / mtotal, 100. * mcorrect / mtotal
        ml1_mean, ml1_std = np.mean(ml1s), np.std(ml1s)
        ml2_mean, ml2_std = np.mean(ml2s), np.std(ml2s)
        mkl_mean, mkl_std = np.mean(mkls), np.std(mkls)
        print('[{}] [VIC] Loss = {:.4f}\tTest-acc = {:.2f}'
              '\tL1 = {:.2f}\tL2 = {:.2f}\tKL = {:.2f}'.format(self.call_count, vtest_loss, vtest_acc,
                                                               ml1_mean, ml2_mean, mkl_mean))

        # Logs
        with open(self.log_path, 'a') as af:
            test_cols = [self.run_id, epoch, 'test', test_loss, test_acc, self.best_test_acc,
                         l1_mean, l1_std, l2_mean, l2_std, kl_mean, kl_std]
            af.write('\t'.join([str(c) for c in test_cols]) + '\n')
            test_cols = [self.run_id, epoch, 'test_vic', vtest_loss, vtest_acc, -1,
                         ml1_mean, ml1_std, ml2_mean, ml2_std, mkl_mean, mkl_std]
            af.write('\t'.join([str(c) for c in test_cols]) + '\n')

        # Dump queries
        query_out_path = osp.join(self.out_path, 'queries.pickle')
        with open(query_out_path, 'wb') as wf:
            pickle.dump(self.queries, wf)
