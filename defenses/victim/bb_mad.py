import argparse
import os.path as osp
import os
import json
import pickle
import itertools
import time

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

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"


class MAD(Blackbox):
    def __init__(self, epsilon=None, optim='linesearch', model_adv_proxy=None, max_grad_layer=None, ydist='l1',
                 oracle='extreme', disable_jacobian=False, objmax=False, out_path=None, log_prefix='', *args, **kwargs):
        super().__init__(*args, **kwargs)
        print('=> MAD ({})'.format([self.dataset_name, epsilon, optim, ydist, oracle]))

        self.epsilon = epsilon
        self.out_path = out_path
        self.disable_jacobian = bool(disable_jacobian)
        if self.disable_jacobian:
            print('')
            print('!!!WARNING!!! Using G = eye(K)')
            print('')

        self.objmax = bool(objmax)

        # Victim's assumption of adversary's model
        print('Proxy for F_A = ', model_adv_proxy)
        if model_adv_proxy is not None:
            if osp.isdir(model_adv_proxy):
                model_adv_proxy_params = osp.join(model_adv_proxy, 'params.json')
                model_adv_proxy = osp.join(model_adv_proxy, 'checkpoint.pth.tar')
                with open(model_adv_proxy_params, 'r') as rf:
                    proxy_params = json.load(rf)
                    model_adv_proxy_arch = proxy_params['model_arch']
                print('Loading proxy ({}) parameters: {}'.format(model_adv_proxy_arch, model_adv_proxy))
            assert osp.exists(model_adv_proxy), 'Does not exist: {}'.format(model_adv_proxy)
            self.model_adv_proxy = zoo.get_net(model_adv_proxy_arch, self.modelfamily, pretrained=model_adv_proxy,
                                               num_classes=self.num_classes)
            self.model_adv_proxy = self.model_adv_proxy.to(self.device)
        else:
            self.model_adv_proxy = self.model

        # To compute stats
        self.dataset = datasets.__dict__[self.dataset_name]
        self.modelfamily = datasets.dataset_to_modelfamily[self.dataset_name]
        self.train_transform = datasets.modelfamily_to_transforms[self.modelfamily]['train']
        self.test_transform = datasets.modelfamily_to_transforms[self.modelfamily]['test']
        self.testset = self.dataset(train=False, transform=self.test_transform)

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
        self.log_path = osp.join(out_path, 'distance{}.log.tsv'.format(log_prefix))
        if not osp.exists(self.log_path):
            with open(self.log_path, 'w') as wf:
                columns = ['call_count', 'l1_mean', 'l1_std', 'l2_mean', 'l2_std', 'kl_mean', 'kl_std']
                wf.write('\t'.join(columns) + '\n')

        self.jacobian_times = []

    @staticmethod
    def compute_jacobian_nll(x, model_adv_proxy, device=torch.device('cuda'), K=None, max_grad_layer=None):
        assert x.shape[0] == 1, 'Does not support batching'
        x = x.to(device)

        # Determine K
        if K is None:
            with torch.no_grad():
                z_a = model_adv_proxy(x)
            _, K = z_a.shape

        # ---------- Precompute G (k x d matrix): where each row represents gradients w.r.t NLL at y_gt = k
        G = []
        z_a = model_adv_proxy(x)
        nlls = -F.log_softmax(z_a, dim=1).mean(dim=0)  # NLL over K classes, Mean over rows
        assert len(nlls) == K
        for k in range(K):
            nll_k = nlls[k]

            _params = [p for p in model_adv_proxy.parameters()]
            w_idx = -2  # Default to FC layer
            # Manually compute gradient only on the required parameters prevents backprop-ing through entire network
            # This is significantly quicker
            # Verified and compared the below with nll_k.backward(retain_graph=True)
            grads, *_ = torch.autograd.grad(nll_k, _params[w_idx], retain_graph=True)
            G.append(grads.flatten().clone())

        G = torch.stack(G).to(device)

        return G

    @staticmethod
    def calc_objective(ytilde, y, G):
        K, D = G.shape
        assert ytilde.shape == y.shape == torch.Size([K, ]), 'Does not support batching'

        with torch.no_grad():
            u = torch.matmul(G.t(), ytilde)
            u = u / u.norm()

            v = torch.matmul(G.t(), y)
            v = v / v.norm()

            objval = (u - v).norm() ** 2

        return objval

    @staticmethod
    def calc_objective_batched(ytilde, y, G):
        K, D = G.shape
        _K, B = ytilde.shape
        assert ytilde.size() == y.size() == torch.Size([K, B]), 'Failed: {} == {} == {}'.format(ytilde.size(), y.size(),
                                                                                                torch.Size([K, B]))

        with torch.no_grad():
            u = torch.matmul(G.t(), ytilde)
            u = u / u.norm(dim=0)

            v = torch.matmul(G.t(), y)
            v = v / v.norm(dim=0)

            objvals = (u - v).norm(dim=0) ** 2

        return objvals

    @staticmethod
    def calc_objective_numpy(ytilde, y, G):
        K, D = G.shape
        assert ytilde.shape == y.shape == torch.Size([K, ]), 'Does not support batching'

        u = G.T @ ytilde
        u /= np.linalg.norm(u)

        v = G.T @ y
        v /= np.linalg.norm(v)

        objval = np.linalg.norm(u - v) ** 2

        return objval

    @staticmethod
    def calc_objective_numpy_batched(ytilde, y, G):
        K, D = G.shape
        _K, N = ytilde.shape
        assert ytilde.shape == y.shape == torch.Size([K, N]), 'Does not support batching'

        u = np.matmul(G.T, ytilde)
        u /= np.linalg.norm(u, axis=0)

        v = np.matmul(G.T, y)
        v /= np.linalg.norm(v, axis=0)

        objvals = np.linalg.norm(u - v, axis=0) ** 2

        return objvals

    @staticmethod
    def calc_surrogate_objective(ytilde, y, G):
        K, D = G.shape
        assert ytilde.shape == y.shape == torch.Size([K, ]), 'ytilde = {}\ty = {}'.format(ytilde.shape, y.shape)

        with torch.no_grad():
            u = torch.matmul(G.t(), ytilde)
            v = torch.matmul(G.t(), y)

            objval = (u - v).norm() ** 2

        return objval

    @staticmethod
    def calc_surrogate_objective_batched(ytilde, y, G):
        K, D = G.shape
        _K, B = ytilde.shape
        assert ytilde.size() == y.size() == torch.Size([K, B]), 'Failed: {} == {} == {}'.format(ytilde.size(), y.size(),
                                                                                                torch.Size([K, B]))

        with torch.no_grad():
            u = torch.matmul(G.t(), ytilde)
            v = torch.matmul(G.t(), y)
            objvals = (u - v).norm(dim=0) ** 2

        return objvals

    @staticmethod
    def calc_surrogate_objective_numpy(ytilde, y, G):
        K, D = G.shape
        assert ytilde.shape == y.shape == torch.Size([K, ]), 'Does not support batching'

        u = G.T @ ytilde
        v = G.T @ y

        objval = np.linalg.norm(u - v) ** 2

        return objval

    @staticmethod
    def calc_surrogate_objective_numpy_batched(ytilde, y, G):
        K, D = G.shape
        assert ytilde.shape == y.shape == torch.Size([K, ]), 'Does not support batching'

        u = np.matmul(G.T, ytilde)
        v = np.matmul(G.T, y)
        objvals = np.linalg.norm(u - v, axis=0) ** 2

        return objvals

    @staticmethod
    def oracle_extreme(G, y, max_over_obj=False):
        K, D = G.shape
        assert y.shape == torch.Size([K, ]), 'Does not support batching'

        argmax_k = -1
        argmax_val = -1.

        for k in range(K):
            yk = torch.zeros_like(y)
            yk[k] = 1.

            if max_over_obj:
                kval = MAD.calc_objective(yk, y, G)
            else:
                kval = MAD.calc_surrogate_objective(yk, y, G)
            if kval > argmax_val:
                argmax_val = kval
                argmax_k = k

        ystar = torch.zeros_like(y)
        ystar[argmax_k] = 1.

        return ystar, argmax_val

    @staticmethod
    def oracle_argmax_preserving(G, y, max_over_obj=False):
        K, D = G.shape
        assert y.shape == torch.Size([K, ]), 'Does not support batching'

        if K > 10:
            # return MAD.oracle_argmax_preserving_approx(G, y, max_over_obj)
            return MAD.oracle_argmax_preserving_approx_gpu(G, y, max_over_obj)

        max_k = y.argmax()
        G_np = G.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()

        ystar = None
        max_val = -1.

        # Generate a set of 1-hot encoded vertices
        # This command produces vertex sets e.g., for K=3: 000, 001, 010, 011, ..., 111
        # Idea is to distribute prob. mass equally over vertices set to 1
        vertices = np.asarray(list(itertools.product([0, 1], repeat=K)), dtype=np.float32)
        # Select vertices where k-th vertex = 1
        vertices = vertices[vertices[:, max_k] > 0]
        # Iterate over these vertices to find argmax k
        for y_extreme in vertices:
            # y_extreme is a K-dim 1-hot encoded vector e.g., [1 0 1 0]
            # Upweigh k-th label by epsilon to maintain argmax label
            y_extreme[max_k] += 1e-5
            # Convert to prob vector
            y_extreme = y_extreme / y_extreme.sum()

            # Doing this on CPU is much faster (I guess this is because we don't need a mem transfer each iteration)
            if max_over_obj:
                kval = MAD.calc_objective_numpy(y_extreme, y_np, G_np)
            else:
                kval = MAD.calc_surrogate_objective_numpy(y_extreme, y_np, G_np)

            if kval > max_val:
                max_val = kval
                ystar = y_extreme

        ystar = torch.tensor(ystar).to(G.device)
        assert ystar.argmax() == y.argmax()

        return ystar, max_val

    @staticmethod
    def oracle_argmax_preserving_approx_gpu(G, y, max_over_obj=False, max_iters=1024):
        K, D = G.shape
        assert y.shape == torch.Size([K, ]), 'Does not support batching'

        max_k = y.argmax().item()
        G_np = G.detach()
        y_np = y.detach().clone()

        # To prevent underflow
        y_np += 1e-8
        y_np /= y_np.sum()

        ystar = None
        max_val = -1.
        niters = 0.

        # By default, we have (K-1)! vertices -- this does not scale when K is large (e.g., CIFAR100).
        # So, perform the search heuristically.
        # Search strategy used:
        #   a. find k which maximizes objective
        #   b. fix k
        #   c. repeat
        fixed_verts = [max_k, ]  # Grow this set

        while niters < max_iters:
            y_prev_extreme = torch.zeros(K)
            y_prev_extreme[fixed_verts] = 1.

            # Find the next vertex extreme
            k_list = np.array(sorted((set(range(K)) - set(fixed_verts))), dtype=int)
            y_extreme_batch = []
            for i, k in enumerate(k_list):
                y_extreme = y_prev_extreme.clone().detach()
                # y_extreme is a K-dim 1-hot encoded vector e.g., [1 0 1 0]
                y_extreme[k] = 1.

                # Upweigh k-th label by epsilon to maintain argmax label
                y_extreme[max_k] += 1e-5
                # Convert to prob vector
                y_extreme = y_extreme / y_extreme.sum()

                y_extreme_batch.append(y_extreme)

            y_extreme_batch = torch.stack(y_extreme_batch).transpose(0, 1).to(G_np.device)
            assert y_extreme_batch.size() == torch.Size([K, len(k_list)]), '{} != {}'.format(y_extreme_batch.size(),
                                                                                             (K, len(k_list)))
            B = y_extreme_batch.size(1)

            y_np_batch = torch.stack([y_np.clone().detach() for i in range(B)]).transpose(0, 1)

            kvals = MAD.calc_objective_batched(y_extreme_batch, y_np_batch, G_np)

            max_i = kvals.argmax().item()
            max_k_val = kvals.max().item()

            if max_k_val > max_val:
                max_val = max_k_val
                ystar = y_extreme_batch[:, max_i]

            next_k = k_list[max_i]
            fixed_verts.append(next_k)

            niters += B

        try:
            ystar = ystar.clone().detach()
        except AttributeError:
            import ipdb;
            ipdb.set_trace()
        assert ystar.argmax() == y.argmax()

        return ystar, max_val

    @staticmethod
    def oracle_argmax_preserving_approx(G, y, max_over_obj=False, max_iters=1024):
        K, D = G.shape
        assert y.shape == torch.Size([K, ]), 'Does not support batching'

        max_k = y.argmax()
        G_np = G.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()

        ystar = None
        max_val = -1.
        niters = 0.

        # By default, we have (K-1)! vertices -- this does not scale when K is large (e.g., CIFAR100).
        # So, perform the search heuristically.
        # Search strategy used:
        #   a. find k which maximizes objective
        #   b. fix k
        #   c. repeat
        fixed_verts = np.array([max_k, ], dtype=int)  # Grow this set
        while niters < max_iters:
            y_prev_extreme = np.zeros((K,), dtype=np.float32)
            y_prev_extreme[fixed_verts] = 1.

            # Find the next vertex extreme
            k_list = np.array(sorted((set(range(K)) - set(fixed_verts))), dtype=int)
            y_extreme_batch = []
            for i, k in enumerate(k_list):
                y_extreme = y_prev_extreme.copy()
                # y_extreme is a K-dim 1-hot encoded vector e.g., [1 0 1 0]
                y_extreme[k] = 1.

                # Upweigh k-th label by epsilon to maintain argmax label
                y_extreme[max_k] += 1e-5
                # Convert to prob vector
                y_extreme = y_extreme / y_extreme.sum()

                y_extreme_batch.append(y_extreme)

            y_extreme_batch = np.array(y_extreme_batch).T
            assert y_extreme_batch.shape == (K, len(k_list)), '{} != {}'.format(y_extreme_batch.shape, (K, len(k_list)))
            B = y_extreme_batch.shape[1]

            y_np_batch = np.stack([y_np.copy() for i in range(B)]).T.astype(np.float32)

            kvals = MAD.calc_objective_numpy_batched(y_extreme_batch, y_np_batch, G_np)

            max_i = np.argmax(kvals)
            max_k_val = np.max(kvals)

            if max_k_val > max_val:
                max_val = max_k_val
                ystar = y_extreme_batch[:, max_i]

            next_k = k_list[max_i]
            fixed_verts = np.concatenate((fixed_verts, [next_k, ]), axis=0)

            niters += B

        ystar = torch.tensor(ystar).to(G.device)
        assert ystar.argmax() == y.argmax()

        return ystar, max_val

    @staticmethod
    def oracle_rand(G, y):
        K, D = G.shape
        assert y.shape == torch.Size([K, ]), 'Does not support batching'

        rand_k = np.random.randint(low=0, high=K)

        ystar = torch.zeros_like(y)
        ystar[rand_k] = 1.
        return ystar, torch.tensor(-1.)

    @staticmethod
    def oracle_argmin(G, y):
        K, D = G.shape
        assert y.shape == torch.Size([K, ]), 'Does not support batching'

        argmin_k = y.argmin().item()
        ystar = torch.zeros_like(y)

        ystar[argmin_k] = 1.
        return ystar, torch.tensor(-1.)

    @staticmethod
    def calc_distance(y, ytilde, ydist, device=torch.device('cuda')):
        assert y.shape == ytilde.shape, 'y = {}, ytile = {}'.format(y.shape, ytilde.shape)
        assert len(y.shape) == 1, 'Does not support batching'
        assert ydist in ['l1', 'l2', 'kl']

        ytilde = ytilde.to(device)
        if ydist == 'l1':
            return (ytilde - y).norm(p=1)
        elif ydist == 'l2':
            return (ytilde - y).norm(p=2)
        elif ydist == 'kl':
            return F.kl_div(ytilde.log(), y, reduction='sum')
        else:
            raise ValueError('Unrecognized ydist contraint')

    @staticmethod
    def project_ydist_constraint(delta, epsilon, ydist, y=None):
        assert len(delta.shape) == 1, 'Does not support batching'
        assert ydist in ['l1', 'l2', 'kl']

        device = delta.device
        K, = delta.shape

        assert delta.shape == torch.Size([K, ])
        if ydist == 'l1':
            delta_numpy = delta.detach().cpu().numpy()
            delta_projected = euclidean_proj_l1ball(delta_numpy, s=epsilon)
            delta_projected = torch.tensor(delta_projected)
        elif ydist == 'l2':
            delta_projected = epsilon * delta / delta.norm(p=2).clamp(min=epsilon)
        elif ydist == 'kl':
            raise NotImplementedError()
        delta_projected = delta_projected.to(device)
        return delta_projected

    @staticmethod
    def project_simplex_constraint(ytilde):
        assert len(ytilde.shape) == 1, 'Does not support batching'
        K, = ytilde.shape
        device = ytilde.device

        ytilde_numpy = ytilde.detach().cpu().numpy()
        ytilde_projected = euclidean_proj_simplex(ytilde_numpy)
        ytilde_projected = torch.tensor(ytilde_projected)
        ytilde_projected = ytilde_projected.to(device)
        return ytilde_projected

    @staticmethod
    def closed_form_alpha_estimate(y, ystar, ydist, epsilon):
        assert y.shape == ystar.shape, 'y = {}, ystar = {}'.format(y.shape, ystar.shape)
        assert len(y.shape) == 1, 'Does not support batching'
        assert ydist in ['l1', 'l2', 'kl']
        K, = y.shape

        if ydist == 'l1':
            p = 1.
        elif ydist == 'l2':
            p = 2.
        else:
            raise ValueError('Only supported for l1/l2')
        alpha = epsilon / ((y - ystar).norm(p=p) + 1e-7)
        alpha = alpha.clamp(min=0., max=1.)
        return alpha

    @staticmethod
    def linesearch(G, y, ystar, ydist, epsilon, closed_alpha=True):
        """
        Let h(\alpha) = (1 - \alpha) y + \alpha y*
        Compute \alpha* = argmax_{\alpha} h(\alpha)
        s.t.  dist(y, h(\alpha)) <= \epsilon

        :param G:
        :param y:
        :param ystar:
        :return:
        """
        K, D = G.shape
        assert y.shape == ystar.shape == torch.Size([K, ]), 'y = {}, ystar = {}'.format(y.shape, ystar.shape)
        assert ydist in ['l1', 'l2', 'kl']

        # Definition of h
        h = lambda alpha: (1 - alpha) * y + alpha * ystar

        # Short hand for distance function
        dist_func = lambda y1, y2: MAD.calc_distance(y1, y2, ydist)

        if ydist in ['l1', 'l2'] and closed_alpha:
            # ---------- Optimally compute alpha
            alpha = MAD.closed_form_alpha_estimate(y, ystar, ydist, epsilon)
            ytilde = h(alpha)
        else:
            # ---------- Bisection method
            alpha_low, alpha_high = 0., 1.
            h_low, h_high = h(alpha_low), h(alpha_high)

            # Sanity check
            feasible_low = dist_func(y, h_low) <= epsilon
            feasible_high = dist_func(y, h_high) <= epsilon
            assert feasible_low or feasible_high

            if feasible_high:
                # Already feasible. Our work here is done.
                ytilde = h_high
                delta = ytilde - y
                return delta
            else:
                ytilde = h_low

            # Binary Search
            for i in range(15):
                alpha_mid = (alpha_low + alpha_high) / 2.
                h_mid = h(alpha_mid)
                feasible_mid = dist_func(y, h_mid) <= epsilon

                if feasible_mid:
                    alpha_low = alpha_mid
                    ytilde = h_mid
                else:
                    alpha_high = alpha_mid

        delta = ytilde - y
        return delta

    @staticmethod
    def greedy(G, y, ystar):
        NotImplementedError()

    @staticmethod
    def is_in_dist_ball(y, ytilde, ydist, epsilon, tolerance=1e-4):
        assert y.shape == ytilde.shape, 'y = {}, ytile = {}'.format(y.shape, ytilde.shape)
        assert len(y.shape) == 1, 'Does not support batching'
        return (MAD.calc_distance(y, ytilde, ydist) - epsilon).clamp(min=0.) <= tolerance

    @staticmethod
    def is_in_simplex(ytilde, tolerance=1e-4):
        assert len(ytilde.shape) == 1, 'Does not support batching'
        return torch.abs(ytilde.clamp(min=0., max=1.).sum() - 1.) <= tolerance

    @staticmethod
    def projections(G, y, ystar, epsilon, ydist, max_iters=100):
        K, D = G.shape
        assert y.shape == ystar.shape == torch.Size([K, ]), 'y = {}, ystar = {}'.format(y.shape, ystar.shape)
        assert ydist in ['l1', 'l2', 'kl']

        ytilde = ystar
        device = G.device

        for i in range(max_iters):
            # (1) Enforce distance constraint
            delta = ytilde - y
            delta = MAD.project_ydist_constraint(delta, epsilon, ydist).to(device)
            ytilde = y + delta

            # (2) Project back into simplex
            ytilde = MAD.project_simplex_constraint(ytilde).to(device)

            # Break out if constraints are met
            if MAD.is_in_dist_ball(y, ytilde, ydist, epsilon) and MAD.is_in_simplex(ytilde):
                break

        delta = ytilde - y
        return delta

    def calc_delta(self, x, y, debug=False):
        # Jacobians G
        if self.disable_jacobian or self.oracle in ['random', 'argmin']:
            G = torch.eye(self.K).to(self.device)
        else:
            # _start = time.time()
            G = MAD.compute_jacobian_nll(x, self.model_adv_proxy, device=self.device, K=self.K)
            # _end = time.time()
            # self.jacobian_times.append(_end - _start)
            # if np.random.random() < 0.05:
            #     print('mean = {:.6f}\tstd = {:.6f}'.format(np.mean(self.jacobian_times), np.std(self.jacobian_times)))
            # # print(_end - _start)
        if self.D is None:
            self.D = G.shape[1]

        # y* via oracle
        if self.oracle == 'random':
            ystar, ystar_val = self.oracle_rand(G, y)
        elif self.oracle == 'extreme':
            ystar, ystar_val = self.oracle_extreme(G, y, max_over_obj=self.objmax)
        elif self.oracle == 'argmin':
            ystar, ystar_val = self.oracle_argmin(G, y)
        elif self.oracle == 'argmax':
            ystar, ystar_val = self.oracle_argmax_preserving(G, y, max_over_obj=self.objmax)
        else:
            raise ValueError()

        # y* maybe outside the feasible set - project it back
        if self.optim == 'linesearch':
            delta = self.linesearch(G, y, ystar, self.ydist, self.epsilon)
        elif self.optim == 'projections':
            delta = self.projections(G, y, ystar, self.ydist, self.epsilon)
        elif self.optim == 'greedy':
            raise NotImplementedError()
        else:
            raise ValueError()

        # Calc. final objective values
        ytilde = y + delta
        objval = self.calc_objective(ytilde, y, G)
        objval_surrogate = self.calc_surrogate_objective(ytilde, y, G)

        return delta, objval, objval_surrogate

    @staticmethod
    def calc_query_distances(queries):
        l1s, l2s, kls = [], [], []
        for i in range(len(queries)):
            y_v, y_prime, *_ = queries[i]
            y_v, y_prime = torch.tensor(y_v), torch.tensor(y_prime)
            l1s.append((y_v - y_prime).norm(p=1).item())
            l2s.append((y_v - y_prime).norm(p=2).item())
            kls.append(F.kl_div(y_prime.log(), y_v, reduction='sum').item())
        l1_mean, l1_std = np.mean(l1s), np.std(l1s)
        l2_mean, l2_std = np.mean(l2s), np.std(l2s)
        kl_mean, kl_std = np.mean(kls), np.std(kls)

        return l1_mean, l1_std, l2_mean, l2_std, kl_mean, kl_std

    def __call__(self, x):
        TypeCheck.multiple_image_blackbox_input_tensor(x)  # of shape B x C x H x W

        with torch.no_grad():
            x = x.to(self.device)
            z_v = self.model(x)  # Victim's predicted logits
            y_v = F.softmax(z_v, dim=1).detach()
            self.call_count += x.shape[0]

        y_prime = []

        # No batch support yet. So, perturb individually.
        for i in range(x.shape[0]):
            x_i = x[i].unsqueeze(0)
            y_v_i = y_v[i]

            with torch.enable_grad():
                if self.epsilon > 0.:
                    delta_i, objval, sobjval = self.calc_delta(x_i, y_v_i)
                else:
                    delta_i = torch.zeros_like(y_v_i)
                    objval, sobjval = torch.tensor(0.), torch.tensor(0.)

            y_prime_i = y_v_i + delta_i

            # ---------------------- Sanity checks
            # ---------- 1. No NaNs
            assert torch.isnan(delta_i).sum().item() == 0., ' y = {}\n delta = {}'.format(y_v_i, delta_i)
            # ---------- 2. Constraints are met
            if not self.is_in_simplex(y_prime_i):
                print('[WARNING] Simplex contraint failed (i = {})'.format(self.call_count))
            if not self.is_in_dist_ball(y_v_i, y_prime_i, self.ydist, self.epsilon):
                _dist = self.calc_distance(y_v_i, y_prime_i, self.ydist)
                print('[WARNING] Distance contraint failed (i = {}, dist = {:.4f} > {:.4f})'.format(self.call_count,
                                                                                                    _dist,
                                                                                                    self.epsilon))

            self.queries.append((y_v_i.cpu().detach().numpy(), y_prime_i.cpu().detach().numpy(),
                                 objval.cpu().detach().numpy(), sobjval.cpu().detach().numpy()))

            y_prime.append(y_prime_i)

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

        y_prime = torch.stack(y_prime)

        return y_prime
