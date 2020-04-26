import argparse
import os.path as osp
import os
import json
from datetime import datetime
import pickle

import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from knockoff.utils.type_checks import TypeCheck
import knockoff.utils.model as model_utils
import knockoff.models.zoo as zoo
from knockoff import datasets

from defenses.victim import Blackbox, MAD
from .bb_reversesigmoid import ReverseSigmoid

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"


class ReverseSigmoid_WB(Blackbox):
    """
    Implementation of "Defending Against Machine Learning Model Stealing Attacks Using Deceptive Perturbations" Lee
        et al.
    """

    def __init__(self, beta=1.0, gamma=1.0, out_path=None, model_adv=None, attacker_argmax=False, adv_optimizer='sgd',
                 log_prefix='', *args, **kwargs):
        super().__init__(*args, **kwargs)
        print('=> ReverseSigmoid ({})'.format([beta, gamma]))

        assert beta >= 0.
        assert gamma >= 0.

        self.beta = beta
        self.gamma = gamma

        self.attacker_argmax = bool(attacker_argmax)
        if self.attacker_argmax:
            print('')
            print('!!!WARNING!!! Argmax of perturbed probabilities used to train attacker model')
            print('')

        ''' 
                    White-box specific stuff 
                '''
        self.out_path = out_path

        # self.model_adv: Adversary's model
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

        # Track some data for debugging
        self.queries = []  # List of (x_i, y_i, y_i_prime, distance)
        self.run_id = str(datetime.now())
        self.log_path = osp.join(out_path, 'online.log.tsv')
        if not osp.exists(self.log_path):
            with open(self.log_path, 'w') as wf:
                columns = ['run_id', 'epoch', 'split', 'loss', 'accuracy', 'best_accuracy', 'l1_mean', 'l1_std',
                           'l2_mean', 'l2_std', 'kl_mean', 'kl_std']
                wf.write('\t'.join(columns) + '\n')

    @staticmethod
    def reverse_sigmoid(y, beta, gamma):
        """
        Equation (3)
        :param y:
        :return:
        """
        return beta * (ReverseSigmoid.sigmoid(gamma * ReverseSigmoid.inv_sigmoid(y)) - 0.5)

    def compute_noise(self, y_v):
        # Inner term of Equation 4
        y_prime = y_v - ReverseSigmoid.reverse_sigmoid(y_v, self.beta, self.gamma)

        # Sum to 1 normalizer "alpha"
        y_prime /= y_prime.sum(dim=1)[:, None]

        delta = y_prime - y_v

        return delta

    def __call__(self, x):
        TypeCheck.multiple_image_blackbox_input_tensor(x)  # of shape B x C x H x W
        assert x.shape[0] == 1, 'Does not support batching. x.shape = {}'.format(x.shape)

        with torch.no_grad():
            x = x.to(self.device)
            z_v = self.model(x)  # Victim's predicted logits
            y_v = F.softmax(z_v, dim=1)
            self.call_count += x.shape[0]

        delta = self.compute_noise(y_v)

        y_prime = y_v + delta

        if len(y_prime.shape) == 1:
            y_prime.unsqueeze_(0)
        self.queries.append((y_v[0].cpu().detach().numpy(), y_prime[0].cpu().detach().numpy()))

        if self.attacker_argmax:
            y_prime = self.truncate_output(y_prime, topk=1, rounding=0)

        # Train adversary
        self.adv_train_step(x, y_prime)

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
                mdelta = self.compute_noise(my_v)
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
