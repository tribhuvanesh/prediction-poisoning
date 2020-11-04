#!/usr/bin/python
"""This is a short description.
Replace this with a more detailed description of what this file contains.
"""
import argparse
import os.path as osp
import os
import pickle
import json
from datetime import datetime
import re

import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
import torchvision

from knockoff import datasets
import knockoff.utils.transforms as transform_utils
import knockoff.utils.model as model_utils
import knockoff.utils.utils as knockoff_utils
import knockoff.config as cfg
import knockoff.models.zoo as zoo

from defenses.adversary.transfer import parse_defense_kwargs, BBOX_CHOICES
from defenses.victim import *

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"


def make_one_hot(labels, K):
    return torch.zeros(labels.shape[0], K, device=labels.device).scatter(1, labels.unsqueeze(1), 1)


class JacobianAdversary:
    """
    PyTorch implementation of:
    1. (JBDA) "Practical Black-Box Attacks against Machine Learning", Papernot et al., ACCS '17
    2. (JB-{topk, self}) "PRADA: Protecting against DNN Model Stealing Attacks", Juuti et al., Euro S&P '19
    """
    def __init__(self, blackbox, budget, model_adv_name, model_adv_pretrained, modelfamily, seedset, testset, device,
                 out_dir, batch_size=cfg.DEFAULT_BATCH_SIZE, train_epochs=20, kappa=400, tau=None, rho=6, sigma=-1,
                 query_batch_size=1, aug_strategy='jbda', useprobs=True, final_train_epochs=100):
        self.blackbox = blackbox
        self.budget = budget
        self.model_adv_name = model_adv_name
        self.model_adv_pretrained = model_adv_pretrained
        self.model_adv = None
        self.modelfamily = modelfamily
        self.seedset = seedset
        self.testset = testset
        self.batch_size = batch_size
        self.query_batch_size = query_batch_size
        self.testloader = DataLoader(self.testset, batch_size=self.batch_size, pin_memory=True)
        self.train_epochs = train_epochs
        self.final_train_epochs = final_train_epochs
        self.kappa = kappa
        self.tau = tau
        self.rho = rho
        self.sigma = sigma
        self.device = device
        self.out_dir = out_dir
        self.num_classes = len(self.testset.classes)
        assert (aug_strategy in ['jbda', 'jbself']) or 'jbtop' in aug_strategy
        self.aug_strategy = aug_strategy
        self.topk = 0
        if 'jbtop' in aug_strategy:
            # extract k from "jbtop<k>"
            self.topk = int(aug_strategy.replace('jbtop', ''))

        self.accuracies = []  # Track test accuracies over time
        self.useprobs = useprobs

        # -------------------------- Initialize seed data
        print('=> Obtaining predictions over {} seed samples using strategy {}'.format(len(self.seedset),
                                                                                       self.aug_strategy))
        Dx = torch.cat([self.seedset[i][0].unsqueeze(0) for i in range(len(self.seedset))])
        Dy = []

        # Populate Dy
        with torch.no_grad():
            for inputs, in DataLoader(TensorDataset(Dx), batch_size=self.query_batch_size):
                inputs = inputs.to(self.device)
                outputs = blackbox(inputs).cpu()
                if not self.useprobs:
                    labels = torch.argmax(outputs, dim=1)
                    labels_onehot = make_one_hot(labels, outputs.shape[1])
                    outputs = labels_onehot
                Dy.append(outputs)
        # Dy = torch.tensor(Dy)
        Dy = torch.cat(Dy)

        # TensorDataset D
        self.D = TensorDataset(Dx, Dy)

        ### Block memory required for training later on
        model_adv = zoo.get_net(self.model_adv_name, self.modelfamily, self.model_adv_pretrained,
                                num_classes=self.num_classes)
        model_adv = model_adv.to(self.device)
        model_adv = model_utils.train_model(model_adv, self.D, self.out_dir, num_workers=10,
                                            checkpoint_suffix='.{}'.format(self.blackbox.call_count),
                                            device=self.device, epochs=1,
                                            log_interval=500, lr=0.01, momentum=0.9, batch_size=self.batch_size,
                                            lr_gamma=0.1, testset=self.testset,
                                            criterion_train=model_utils.soft_cross_entropy)


    def get_transferset(self):
        """
        :return:
        """
        # for rho_current in range(self.rho):
        rho_current = 0
        while self.blackbox.call_count < self.budget:
            print('=> Beginning substitute epoch {} (|D| = {})'.format(rho_current, len(self.D)))
            # -------------------------- 0. Initialize Model
            model_adv = zoo.get_net(self.model_adv_name, self.modelfamily, self.model_adv_pretrained,
                                    num_classes=self.num_classes)
            model_adv = model_adv.to(self.device)

            # -------------------------- 1. Train model on D
            model_adv = model_utils.train_model(model_adv, self.D, self.out_dir, num_workers=10,
                                                checkpoint_suffix='.{}'.format(self.blackbox.call_count),
                                                device=self.device, epochs=self.train_epochs, log_interval=500, lr=0.1,
                                                momentum=0.9, batch_size=self.batch_size, lr_gamma=0.1,
                                                testset=self.testset, criterion_train=model_utils.soft_cross_entropy)

            # -------------------------- 2. Evaluate model
            # _, acc = model_utils.test_step(model_adv, self.testloader, nn.CrossEntropyLoss(reduction='mean'),
            #                                device=self.device, epoch=rho_current)
            # self.accuracies.append(acc)

            # -------------------------- 3. Jacobian-based data augmentation
            if self.aug_strategy in ['jbda', 'jbself']:
                self.D = self.jacobian_augmentation(model_adv, rho_current)
            elif self.aug_strategy == 'jbtop{}'.format(self.topk):
                self.D = self.jacobian_augmentation_topk(model_adv, rho_current)
            else:
                raise ValueError('Unrecognized augmentation strategy: "{}"'.format(self.aug_strategy))

            # -------------------------- 4. End if necessary
            rho_current += 1
            if (self.blackbox.call_count >= self.budget) or ((self.rho is not None) and (rho_current >= self.rho)):
                print('=> # BB Queries ({}) >= budget ({}). Ending attack.'.format(self.blackbox.call_count,
                                                                                   self.budget))
                model_adv = zoo.get_net(self.model_adv_name, self.modelfamily, self.model_adv_pretrained,
                                        num_classes=self.num_classes)
                model_adv = model_adv.to(self.device)
                model_adv = model_utils.train_model(model_adv, self.D, self.out_dir, num_workers=10,
                                                    checkpoint_suffix='.{}'.format(self.blackbox.call_count),
                                                    device=self.device, epochs=self.final_train_epochs,
                                                    log_interval=500, lr=0.01, momentum=0.9, batch_size=self.batch_size,
                                                    lr_gamma=0.1, testset=self.testset,
                                                    criterion_train=model_utils.soft_cross_entropy)
                break

            print()

        return self.D, model_adv

    @staticmethod
    def rand_sample(D, kappa):
        # Note: the paper does reservoir sampling to select kappa elements from D. Since |D| in our case cannot grow
        # larger than main memory size, we randomly sample for simplicity. In either case, each element is drawn with a
        # probability kappa/|D|
        n = len(D)
        idxs = np.arange(n)
        sampled_idxs = np.random.choice(idxs, size=kappa, replace=False)
        mask = np.zeros_like(idxs).astype(bool)
        mask[sampled_idxs] = True
        D_sampled = TensorDataset(D.tensors[0][mask], D.tensors[1][mask])
        return D_sampled

    def jacobian_augmentation(self, model_adv, rho_current, step_size=0.1):
        if (self.kappa is not None) and (rho_current >= self.sigma):
            D_sampled = self.rand_sample(self.D, self.kappa)
        else:
            D_sampled = self.D

        if len(D_sampled) + self.blackbox.call_count >= self.budget:
            # Reduce augmented data size to match query budget
            nqueries_remaining = self.budget - self.blackbox.call_count
            assert nqueries_remaining >= 0
            print('=> Reducing augmented input size ({} -> {}) to stay within query budget.'.format(
                D_sampled.tensors[0].shape[0], nqueries_remaining))
            D_sampled = TensorDataset(D_sampled.tensors[0][:nqueries_remaining],
                                      D_sampled.tensors[1][:nqueries_remaining])

        if self.tau is not None:
            step_size = step_size * ((-1) ** (round(rho_current / self.tau)))

        print('=> Augmentation set size = {} (|D| = {}, B = {})'.format(len(D_sampled), len(self.D),
                                                                        self.blackbox.call_count))
        loader = DataLoader(D_sampled, batch_size=self.query_batch_size, shuffle=False)
        for i, (X, Y) in enumerate(loader):
            start_idx = i * self.query_batch_size
            end_idx = min(start_idx + self.query_batch_size, len(D_sampled))
            # A simple check to ensure we are overwriting the correct input-outputs
            assert Y.sum() == D_sampled.tensors[1][start_idx:end_idx].sum(), '[{}] {} != {}'.format(i, Y.sum(),
                                                                                                    D_sampled.tensors[
                                                                                                        1][
                                                                                                    start_idx:end_idx].sum())
            assert X.sum() == D_sampled.tensors[0][start_idx:end_idx].sum(), '[{}] {} != {}'.format(i, X.sum(),
                                                                                                    D_sampled.tensors[
                                                                                                        0][
                                                                                                    start_idx:end_idx].sum())

            # Get augmented inputs
            X, Y = X.to(self.device), Y.to(self.device)
            delta_i = self.fgsm_untargeted(model_adv, X, Y.argmax(dim=1), device=self.device, epsilon=step_size)
            # Get corrensponding outputs from blackbox
            if self.aug_strategy == 'jbda':
                Y_i = self.blackbox(X + delta_i)
            elif self.aug_strategy == 'jbself':
                Y_i = self.blackbox(X - delta_i)
            else:
                raise ValueError('Unrecognized augmentation strategy {}'.format(self.aug_strategy))

            if not self.useprobs:
                labels = torch.argmax(Y_i, dim=1)
                labels_onehot = make_one_hot(labels, Y_i.shape[1])
                Y_i = labels_onehot

            # Rewrite D_sampled
            D_sampled.tensors[0][start_idx:end_idx] = (X + delta_i).detach().cpu()
            D_sampled.tensors[1][start_idx:end_idx] = Y_i.detach().cpu()

        Dx_augmented = torch.cat([self.D.tensors[0], D_sampled.tensors[0]])
        Dy_augmented = torch.cat([self.D.tensors[1], D_sampled.tensors[1]])
        D_augmented = TensorDataset(Dx_augmented, Dy_augmented)

        return D_augmented

    def jacobian_augmentation_topk(self, model_adv, rho_current, step_size=0.1):
        if (self.kappa is not None) and (rho_current >= self.sigma):
            D_sampled = self.rand_sample(self.D, self.kappa)
        else:
            D_sampled = self.D

        if (len(D_sampled) * self.topk) + self.blackbox.call_count >= self.budget:
            # Reduce augmented data size to match query budget
            nqueries_remaining = self.budget - self.blackbox.call_count
            nqueries_remaining /= 3.
            nqueries_remaining = int(np.ceil(nqueries_remaining))
            assert nqueries_remaining >= 0
            print('=> Reducing augmented input size ({}*{} -> {}*{}={}) to stay within query budget.'.format(
                D_sampled.tensors[0].shape[0], self.topk, nqueries_remaining, self.topk,
                nqueries_remaining * self.topk))
            D_sampled = TensorDataset(D_sampled.tensors[0][:nqueries_remaining],
                                      D_sampled.tensors[1][:nqueries_remaining])

        if self.tau is not None:
            step_size = step_size * ((-1) ** (round(rho_current / self.tau)))

        print('=> Augmentation set size = {} (|D| = {}, B = {})'.format(len(D_sampled), len(self.D),
                                                                        self.blackbox.call_count))
        loader = DataLoader(D_sampled, batch_size=1, shuffle=False)
        X_aug = []
        Y_aug = []
        for i, (X, Y) in enumerate(loader):
            assert X.shape[0] == Y.shape[0] == 1, 'Only supports batch_size = 1'
            X, Y = X.to(self.device), Y.to(self.device)
            Y_pred = model_adv(X)[0]
            Y_pred_sorted = torch.argsort(Y_pred, descending=True)
            Y_pred_sorted = Y_pred_sorted[Y_pred_sorted != Y[0].argmax()]  # Remove gt class

            for c in Y_pred_sorted[:self.topk]:
                delta_i = self.pgd_linf_targ(model_adv, X, Y.argmax(dim=1), c, epsilon=step_size, alpha=0.01,
                                             device=self.device)
                Y_i = self.blackbox(X + delta_i)

                if not self.useprobs:
                    labels = torch.argmax(Y_i, dim=1)
                    labels_onehot = make_one_hot(labels, Y_i.shape[1])
                    Y_i = labels_onehot

                X_aug.append((X + delta_i).detach().cpu().clone())
                Y_aug.append(Y_i.detach().cpu().clone())

            if self.blackbox.call_count >= self.budget:
                break

        X_aug = torch.cat(X_aug, dim=0)
        Y_aug = torch.cat(Y_aug, dim=0)

        Dx_augmented = torch.cat([self.D.tensors[0], X_aug])[:self.budget]
        Dy_augmented = torch.cat([self.D.tensors[1], Y_aug])[:self.budget]
        D_augmented = TensorDataset(Dx_augmented, Dy_augmented)

        return D_augmented

    @staticmethod
    def fgsm_untargeted(model, inputs, targets, epsilon, device, clamp=(0.0, 1.0)):
        if epsilon == 0:
            return torch.zeros_like(inputs)

        with torch.enable_grad():
            inputs, targets = inputs.to(device), targets.to(device)
            inputs.requires_grad = True

            out = model(inputs)
            loss = F.cross_entropy(out, targets)
            loss.backward()

            delta = epsilon * inputs.grad.detach().sign().to(device)

            delta.data = torch.min(torch.max(delta, -inputs),
                                   1 - inputs)  # clip samples+perturbation to [0,1]

            return delta

    @staticmethod
    def pgd_linf_targ(model, inputs, targets, y_targ, epsilon, alpha, device, num_iter=8):
        """ Construct targeted adversarial examples on the examples X"""
        if epsilon == 0:
            return torch.zeros_like(inputs)

        with torch.enable_grad():
            inputs = inputs.to(device)
            delta = torch.zeros_like(inputs, requires_grad=True).to(device)
            for t in range(num_iter):
                yp = model(inputs + delta)
                # loss = (yp[:, y_targ] - yp.gather(1, targets[:, None])[:, 0]).sum()
                loss = yp[:, y_targ].sum()
                loss.backward()
                delta.data = (delta + alpha * delta.grad.detach().sign()).clamp(-epsilon, epsilon)
                delta.grad.zero_()
            return delta.detach()


def main():
    parser = argparse.ArgumentParser(description='Jacobian Model Stealing Attack')
    parser.add_argument('policy', metavar='PI', type=str, help='Policy to use while training')
    parser.add_argument('victim_model_dir', metavar='PATH', type=str,
                        help='Path to victim model. Should contain files "model_best.pth.tar" and "params.json"')
    parser.add_argument('defense', metavar='TYPE', type=str, help='Type of defense to use',
                        choices=BBOX_CHOICES, default='none')
    parser.add_argument('defense_args', metavar='STR', type=str, help='Blackbox arguments in format "k1:v1,k2:v2,..."')
    parser.add_argument('--model_adv', metavar='STR', type=str, help='Model arch of F_A', default=None)
    parser.add_argument('--pretrained', metavar='STR', type=str, help='Assumption of F_A', default=None)
    parser.add_argument('--out_dir', metavar='PATH', type=str,
                        help='Destination directory to store transfer set', required=True)
    parser.add_argument('--testset', metavar='TYPE', type=str, help='Blackbox testset (P_V(X))', required=True)
    parser.add_argument('--batch_size', metavar='TYPE', type=int, help='Batch size of queries',
                        default=cfg.DEFAULT_BATCH_SIZE)
    # ----------- Params for Jacobian-based augmentation
    parser.add_argument('--budget', metavar='N', type=int, help='Query limit to blackbox', default=10000)
    parser.add_argument('--queryset', metavar='TYPE', type=str, help='Data for seed images', required=True)
    parser.add_argument('--seedsize', metavar='N', type=int, help='Size of seed set', default=100)
    parser.add_argument('--rho', metavar='N', type=int, help='# Data Augmentation Steps', default=None)
    parser.add_argument('--sigma', metavar='N', type=int, help='Reservoir sampling beyond these many epochs', default=3)
    parser.add_argument('--kappa', metavar='N', type=int, help='Size of reservoir', default=None)
    parser.add_argument('--tau', metavar='N', type=int,
                        help='Iteration period after which step size is multiplied by -1', default=5)
    parser.add_argument('--train_epochs', metavar='N', type=int, help='# Epochs to train model', default=20)
    # ----------- Other params
    parser.add_argument('-d', '--device_id', metavar='D', type=int, help='Device id', default=0)
    parser.add_argument('-w', '--nworkers', metavar='N', type=int, help='# Worker threads to load data', default=10)
    parser.add_argument('--train_transform', action='store_true', help='Perform data augmentation', default=False)
    args = parser.parse_args()
    params = vars(args)

    out_path = params['out_dir']
    knockoff_utils.create_dir(out_path)

    np.random.seed(cfg.DEFAULT_SEED)
    torch.manual_seed(cfg.DEFAULT_SEED)
    torch.cuda.manual_seed(cfg.DEFAULT_SEED)

    if params['device_id'] >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(params['device_id'])
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # ----------- Set up queryset
    queryset_name = params['queryset']
    valid_datasets = datasets.__dict__.keys()
    if queryset_name not in valid_datasets:
        raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
    modelfamily = datasets.dataset_to_modelfamily[queryset_name]
    transform_type = 'train' if params['train_transform'] else 'test'
    if params['train_transform']:
        print('=> Using data augmentation while querying')
    transform = datasets.modelfamily_to_transforms[modelfamily][transform_type]
    queryset = datasets.__dict__[queryset_name](train=True, transform=transform)

    # Use a subset of queryset
    subset_idxs = np.random.choice(range(len(queryset)), size=params['seedsize'], replace=False)
    seedset = Subset(queryset, subset_idxs)

    # ----------- Set up testset
    testset_name = params['testset']
    if testset_name not in valid_datasets:
        raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
    modelfamily = datasets.dataset_to_modelfamily[testset_name]
    transform_type = 'test'
    transform = datasets.modelfamily_to_transforms[modelfamily][transform_type]
    testset = datasets.__dict__[testset_name](train=False, transform=transform)
    num_classes = len(testset.classes)

    # ----------- Initialize blackbox
    blackbox_dir = params['victim_model_dir']
    defense_type = params['defense']
    if defense_type == 'rand_noise':
        BB = RandomNoise
    elif defense_type == 'rand_noise_wb':
        BB = RandomNoise_WB
    elif defense_type == 'mad':
        BB = MAD
    elif defense_type == 'mad_wb':
        BB = MAD_WB
    elif defense_type == 'reverse_sigmoid':
        BB = ReverseSigmoid
    elif defense_type == 'reverse_sigmoid_wb':
        BB = ReverseSigmoid_WB
    elif defense_type in ['none', 'topk', 'rounding']:
        BB = Blackbox
    else:
        raise ValueError('Unrecognized blackbox type')
    defense_kwargs = parse_defense_kwargs(params['defense_args'])
    defense_kwargs['log_prefix'] = 'transfer'
    print('=> Initializing BBox with defense {} and arguments: {}'.format(defense_type, defense_kwargs))
    blackbox = BB.from_modeldir(blackbox_dir, device, **defense_kwargs)

    for k, v in defense_kwargs.items():
        params[k] = v

    # ----------- Initialize adversary
    budget = params['budget']
    model_adv_name = params['model_adv']
    model_adv_pretrained = params['pretrained']
    train_epochs = params['train_epochs']
    batch_size = params['batch_size']
    kappa = params['kappa']
    tau = params['tau']
    rho = params['rho']
    sigma = params['sigma']
    policy = params['policy']
    adversary = JacobianAdversary(blackbox, budget, model_adv_name, model_adv_pretrained, modelfamily, seedset,
                                  testset, device, out_path, batch_size=batch_size,
                                  train_epochs=train_epochs, kappa=kappa, tau=tau, rho=rho,
                                  sigma=sigma, aug_strategy=policy)

    print('=> constructing transfer set...')
    transferset, model_adv = adversary.get_transferset()
    # import ipdb; ipdb.set_trace()
    # These can be massive (>30G) -- skip it for now
    # transfer_out_path = osp.join(out_path, 'transferset.pickle')
    # with open(transfer_out_path, 'wb') as wf:
    #     pickle.dump(transferset, wf, protocol=pickle.HIGHEST_PROTOCOL)
    # print('=> transfer set ({} samples) written to: {}'.format(len(transferset), transfer_out_path))

    # Store arguments
    params['created_on'] = str(datetime.now())
    params_out_path = osp.join(out_path, 'params_transfer.json')
    with open(params_out_path, 'w') as jf:
        json.dump(params, jf, indent=True)


if __name__ == '__main__':
    main()
