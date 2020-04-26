import argparse
import os.path as osp
import os
import json
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

from defenses.victim.blackbox import Blackbox
from .bb_mad import MAD   # euclidean_proj_l1ball, euclidean_proj_simplex, is_in_dist_ball, is_in_simplex
from .bb_reversesigmoid import ReverseSigmoid

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"


class RandomNoise(Blackbox):
    def __init__(self, epsilon_z=None, dist_z=None, strat='uniform', out_path=None, log_prefix='', *args, **kwargs):
        super().__init__(*args, **kwargs)
        print('=> RandomNoise ({})'.format([epsilon_z, dist_z]))

        self.epsilonz = epsilon_z
        self.out_path = out_path

        assert dist_z in ['l1', 'l2']
        self.ydist = dist_z

        assert strat in ['uniform', 'gaussian']
        self.strat = strat

        # Track some data for debugging
        self.queries = []  # List of (x_i, y_i, y_i_prime, distance)
        self.log_path = osp.join(out_path, 'distance{}.log.tsv'.format(log_prefix))
        if not osp.exists(self.log_path):
            with open(self.log_path, 'w') as wf:
                columns = ['call_count', 'l1_mean', 'l1_std', 'l2_mean', 'l2_std', 'kl_mean', 'kl_std']
                wf.write('\t'.join(columns) + '\n')

    @staticmethod
    def make_one_hot(labels, K):
        return torch.zeros(labels.shape[0], K, device=labels.device).scatter(1, labels.unsqueeze(1), 1)

    @staticmethod
    def compute_noise(y, strat, epsilon_z, zdist):
        """
        Compute noise in the logit space (inverse sigmoid)
        :param y:
        :return:
        """
        z = ReverseSigmoid.inv_sigmoid(y)
        N, K = z.shape
        if strat == 'uniform':
            deltaz = torch.rand_like(z)

            # Norm of expected value of this distribution (|| E_{v ~ Unif[0, high]^K}[v] ||_p) is:
            #       \sqrt[p]{K} * (high=1)/2
            # Setting this to epsilon and solving for high', we get: high' = (2 * epsilon) / \sqrt[p]{K}
            # By drawing a k-dim vector v uniformly in the range [0, high'], we get || E[v] ||_p = epsilon
            if zdist in ['l1', 'l2']:
                p = int(zdist[-1])
                mult = (2 * epsilon_z) / np.power(K, 1. / p)
                # Rescale to [0, high']
                deltaz *= mult
        elif strat == 'gaussian':
            deltaz = torch.randn_like(z)
        else:
            raise ValueError('Unrecognized argument')

        for i in range(N):
            # Project each delta back into ydist space
            # print('Before: {} (norm-{} = {})'.format(deltaz[i], zdist, deltaz[i].norm(p=int(zdist[-1]))))
            deltaz[i] = MAD.project_ydist_constraint(deltaz[i], epsilon_z, zdist)
            # print('After: {} (norm-{} = {})'.format(deltaz[i], zdist, deltaz[i].norm(p=int(zdist[-1]))))
            # print()

        ztilde = z + deltaz
        ytilde = torch.sigmoid(ztilde)
        if len(ytilde.shape) > 1:
            ytilde /= ytilde.sum(dim=1)[:, None]
        else:
            ytilde = ytilde / ytilde.sum()
        delta = ytilde - y

        return delta

    def __call__(self, x):
        TypeCheck.multiple_image_blackbox_input_tensor(x)   # of shape B x C x H x W

        with torch.no_grad():
            x = x.to(self.device)
            z_v = self.model(x)   # Victim's predicted logits
            y_v = F.softmax(z_v, dim=1)
            self.call_count += x.shape[0]

        if self.epsilonz > 0.:
            delta = self.compute_noise(y_v, self.strat, self.epsilonz, self.ydist).to(self.device)
        else:
            delta = torch.zeros_like(y_v)
        y_prime = y_v + delta

        for i in range(x.shape[0]):
            self.queries.append((y_v[i].cpu().detach().numpy(), y_prime[i].cpu().detach().numpy()))

            if (self.call_count + i) % 1000 == 0:
                # Dump queries
                query_out_path = osp.join(self.out_path, 'queries.pickle')
                with open(query_out_path, 'wb') as wf:
                    pickle.dump(self.queries, wf)

                l1_mean, l1_std, l2_mean, l2_std, kl_mean, kl_std = MAD.calc_query_distances(self.queries)

                # Logs
                with open(self.log_path, 'a') as af:
                    test_cols = [self.call_count, l1_mean, l1_std, l2_mean, l2_std, kl_mean, kl_std]
                    af.write('\t'.join([str(c) for c in test_cols]) + '\n')

        return y_prime