from hmac import new
import time
import numpy as np
import argparse
import os
from sklearn.linear_model import LogisticRegression
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
from scipy.optimize import minimize_scalar
import sympy as sp
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from utils import load_features, generate_gaussian, plot_2dgaussian, plot_w_2dgaussian, create_nested_folder
from sgd import stochastic_gradient_descent_algorithm


class Runner():
    def __init__(self, args):
        self.device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')
        self.args = args
        if args.dataset == 'MNIST' or args.dataset == 'MNIST_multiclass':
            self.X_train, self.X_test, self.y_train, self.y_train_onehot, self.y_test = load_features(args)
            self.X_train = self.X_train[:11264]
            self.y_train = self.y_train[:11264]
            self.dim_w = 784
            if args.dataset == 'MNIST':
                self.num_class = 2
            else:
                self.num_class = 10
        elif args.dataset == 'CIFAR10' or args.dataset == 'CIFAR10_multiclass':
            self.X_train, self.X_test, self.y_train, self.y_train_onehot, self.y_test = load_features(args)
            self.X_train = self.X_train[:9856]
            self.y_train = self.y_train[:9856]
            self.dim_w = 512
            if args.dataset == 'CIFAR10':
                self.num_class = 2
            else:
                self.num_class = 10
        # make the norm of x = 1, MNIST naturally satisfys
        self.X_train_norm = self.X_train.norm(dim=1, keepdim=True)
        self.X_train = self.X_train / self.X_train_norm
        self.X_test_norm = self.X_test.norm(dim=1, keepdim=True)
        self.X_test = self.X_test / self.X_test_norm
        self.X_train = self.X_train.to(self.device)
        self.y_train = self.y_train.to(self.device)
    def get_metadata(self):
        # note here the loss function here all times 100 than before
        # num of training data
        self.n = len(self.X_train)
        print('number training data:'+str(self.n))
        # L-smoothness constant
        self.L = 1 / 4 + self.args.lam * self.n
        print('L smooth constant'+str(self.L))
        # m-strongly convex constant
        self.m = self.args.lam * self.n
        print('m strongly convex:'+str(self.m))
        # M-Lipschitz constant
        self.M = self.args.M
        print('M lipschitz constant:'+str(self.M))
        # calculate step size
        #max_eta = min( 1 / self.m, 2 / self.L) 
        self.eta = 2 / self.L
        print('step size eta:'+str(self.eta))
        # calculate RDP delta
        self.delta = 1 / self.n
        print('RDP constant delta:'+str(self.delta))
        # calculate the projection
        self.projection = self.args.projection
        print('weight projection radius: '+str(self.projection))
        # calculate batch size
        self.batch_size = self.args.batch_size
        print('batch size: '+str(self.batch_size))
        # give a fixed index to decide the batch list
        batch_idx = np.arange(self.n)
        np.random.shuffle(batch_idx)
        self.batch_idx = batch_idx
        print('have shuffled batch idx')
        
    def train(self):
        if self.args.search_burnin:
            # this is for full-batch
            if self.args.dataset == 'MNIST':
                sigma_list = [0.005, 0.01, 0.05, 0.1]
                burn_in_list = [1, 10, 20, 50, 100, 150, 200, 300, 500, 750, 1000]
            elif self.args.dataset == 'MNIST_multiclass':
                sigma_list = [0.005, 0.01, 0.05, 0.1]
                burn_in_list = [1, 10, 20, 50, 100, 150, 200, 300, 500, 750, 1000]
            elif self.args.dataset == 'CIFAR10':
                sigma_list = [0.005, 0.01, 0.05, 0.1]
                burn_in_list = [1, 10, 20, 50, 100, 150, 200, 300, 500, 750, 1000]
            _ = self.search_burnin(sigma_list, burn_in_list)
        elif self.args.search_batch:
            batch_list = [64, 128]
            burn_in_list = [1, 10, 20, 50, 100, 150, 200, 300, 500, 750, 1000]
            _ = self.search_batch(burn_in_list, batch_list)
        elif self.args.paint_utility_s:
            num_remove_list = [1, 10, 50, 100, 500, 1000]
            accuracy_scratch_D, mean_time, w_list = self.get_mean_performance(self.X_train, self.y_train, self.args.burn_in, self.args.sigma, None, len_list = 1, return_w = True)
            np.save('./result/LMC/'+str(self.args.dataset)+'/paint_utility_s/learn_scratch_w.npy', w_list)
            np.save('./result/LMC/'+str(self.args.dataset)+'/paint_utility_s/acc_scratch_D.npy', accuracy_scratch_D)
            # calculate K
            epsilon_list = [0.5, 1, 2] # set epsilon = 1
            K_dict, _ = self.search_finetune_step(self.args.sigma, epsilon_list, num_remove_list)
            for epsilon_idx, epsilon in enumerate(epsilon_list):
                K_list = []
                for num_remove in num_remove_list:
                    create_nested_folder('./result/LMC/'+str(self.args.dataset)+'/paint_utility_s/'+str(epsilon)+'/')
                    X_train_removed, y_train_removed = self.get_removed_data(num_remove)
                    accuracy_scratch_Dnew, mean_time = self.get_mean_performance(X_train_removed, y_train_removed, self.args.burn_in, self.args.sigma, None)
                    np.save('./result/LMC/'+str(self.args.dataset)+'/paint_utility_s/'+str(epsilon)+'/acc_scratch_Dnew_remove'+str(num_remove)+'.npy', accuracy_scratch_Dnew)
                    accuracy_finetune, mean_time = self.get_mean_performance(X_train_removed, y_train_removed, K_dict[num_remove][epsilon], self.args.sigma, w_list)
                    np.save('./result/LMC/'+str(self.args.dataset)+'/paint_utility_s/'+str(epsilon)+'/acc_finetune_remove'+str(num_remove)+'.npy', accuracy_finetune)
                    K_list.append(K_dict[num_remove][epsilon])
                np.save('./result/LMC/'+str(self.args.dataset)+'/paint_utility_s/'+str(epsilon)+'/K_list.npy', K_list)
        elif self.args.paint_utility_epsilon:
            epsilon_list = [0.1, 0.5, 1, 2, 5]
            num_remove_list = [1, 50, 100]
            accuracy_scratch_D, mean_time, w_list = self.get_mean_performance(self.X_train, self.y_train, self.args.burn_in, self.args.sigma, None, len_list = 1, return_w = True)
            np.save('./result/LMC/'+str(self.args.dataset)+'/paint_utility_epsilon/w_from_scratch.npy', w_list)
            np.save('./result/LMC/'+str(self.args.dataset)+'/paint_utility_epsilon/acc_scratch_D.npy', accuracy_scratch_D)
            # calculate K
            K_dict, _ = self.search_finetune_step(self.args.sigma, epsilon_list, num_remove_list)
            np.save('./result/LMC/'+str(self.args.dataset)+'/paint_utility_epsilon/K_list.npy', K_dict)
            for remove_idx, num_remove in enumerate(num_remove_list):
                K_list = []
                for epsilon in epsilon_list:
                    X_train_removed, y_train_removed = self.get_removed_data(num_remove_list[remove_idx])
                    accuracy_finetune, mean_time = self.get_mean_performance(X_train_removed, y_train_removed, K_dict[num_remove_list[remove_idx]][epsilon], self.args.sigma, w_list)
                    create_nested_folder('./result/LMC/'+str(self.args.dataset)+'/paint_utility_epsilon/'+str(num_remove)+'/')
                    np.save('./result/LMC/'+str(self.args.dataset)+'/paint_utility_epsilon/'+str(num_remove)+'/acc_finetune_epsilon'+str(epsilon)+'.npy', accuracy_finetune)
                    K_list.append(K_dict[num_remove_list[0]][epsilon])
        elif self.args.paint_unlearning_sigma:
            num_remove_list = [100]
            epsilon_list = [1]
            
            sigma_list = [0.05, 0.1, 0.2, 0.5, 1]
            scratch_acc_list = []
            scratch_unlearn_list = []
            finetune_unlearn_list = []
            epsilon0_list = []
            X_train_removed, y_train_removed = self.get_removed_data(num_remove_list[0])
            for sigma in sigma_list:
                K_dict, alpha_dict = self.search_finetune_step(sigma, epsilon_list, num_remove_list)
                np.save('./result/LMC/'+str(self.args.dataset)+'/paint_unlearning_sigma/K_dict'+str(sigma)+'.npy', K_dict)
                np.save('./result/LMC/'+str(self.args.dataset)+'/paint_unlearning_sigma/alpha_dict'+str(sigma)+'.npy', alpha_dict)
                alpha = alpha_dict[num_remove_list[0]][epsilon_list[0]]
                epsilon0 = self.calculate_epsilon0(alpha, num_remove_list[0], sigma)
                epsilon0_list.append(epsilon0)
                accuracy_scratch_D, mean_time, w_list = self.get_mean_performance(self.X_train, self.y_train, self.args.burn_in, sigma, None, len_list = 1, return_w = True)
                np.save('./result/LMC/'+str(self.args.dataset)+'/paint_unlearning_sigma/'+str(sigma)+'_learn_scratch_w.npy', w_list)
                np.save('./result/LMC/'+str(self.args.dataset)+'/paint_unlearning_sigma/'+str(sigma)+'_acc_scratch_D.npy', accuracy_scratch_D)
                accuracy_scratch_Dnew, mean_time, unlearn_w_list = self.get_mean_performance(X_train_removed, y_train_removed, self.args.burn_in, sigma, None, return_w=True)
                np.save('./result/LMC/'+str(self.args.dataset)+'/paint_unlearning_sigma/'+str(sigma)+'_unlearn_scratch_w.npy', unlearn_w_list)
                np.save('./result/LMC/'+str(self.args.dataset)+'/paint_unlearning_sigma/'+str(sigma)+'_acc_scratch_Dnew.npy', accuracy_scratch_Dnew)
                accuracy_finetune, mean_time = self.get_mean_performance(X_train_removed, y_train_removed, K_dict[num_remove_list[0]][1], sigma, w_list)
                np.save('./result/LMC/'+str(self.args.dataset)+'/paint_unlearning_sigma/'+str(sigma)+'_acc_finetune.npy', accuracy_finetune)
            np.save('./result/LMC/'+str(self.args.dataset)+'/paint_unlearning_sigma/epsilon0.npy', epsilon0_list)
        elif self.args.how_much_retrain == 1:
            sigma_list = [0.05, 0.1, 0.2, 0.5, 1]
            if self.args.dataset == 'MNIST':
                K_list = [1301, 1031, 751, 351, 1]
            elif self.args.dataset =='CIFAR10':
                K_list = [1541, 1251, 951, 521, 151]
            num_remove_list = [100]
            X_train_removed, y_train_removed = self.get_removed_data(num_remove_list[0])
            create_nested_folder('./result/LMC/'+str(self.args.dataset)+'/retrain/')
            for sigma_idx, sigma in enumerate(sigma_list):
                accuracy_scratch_D, mean_time = self.get_mean_performance(X_train_removed, y_train_removed, K_list[sigma_idx], sigma, None, len_list = 1)
                np.save('./result/LMC/'+str(self.args.dataset)+'/retrain/'+str(sigma)+'_acc_scratch_D.npy', accuracy_scratch_D)
                print('sigma:'+str(sigma))
                print('mean acc:'+str(np.mean(accuracy_scratch_D)))
                print('std acc:'+str(np.std(accuracy_scratch_D)))
        else:
            print('check!')

    def get_removed_data(self, num_remove):
        X_train_removed = self.X_train[:-num_remove,:]
        y_train_removed = self.y_train[:-num_remove]
        new_X_train = torch.randn(num_remove, self.dim_w)
        norms = new_X_train.norm(dim=1, keepdim=True)
        new_X_train = new_X_train / norms
        new_X_train = new_X_train.to(self.device)
        new_y_train = torch.randint(0, 2, (1, num_remove)) * 2 - 1
        new_y_train = new_y_train.to(self.device).reshape(-1)
        X_train_removed = torch.cat((X_train_removed, new_X_train), 0)
        y_train_removed = torch.cat((y_train_removed, new_y_train))
        return X_train_removed, y_train_removed
        
    def epsilon_expression(self, K, sigma, eta, C_lsi, alpha, S, M, m, n, delta):
        part_1 = math.exp(- (float(K) * m * float(eta)) / (alpha))
        part_2 = (4 * alpha * float(S)**2 * float(M)**2) / (float(m) * float(sigma)**2 * float(n)**2)
        part_3 = (math.log(1 / float(delta))) / (alpha - 1)
        epsilon = part_1 * part_2 + part_3
        return epsilon
    
    def search_finetune_step(self, sigma, epsilon_list, num_remove_list):
        C_lsi = 2 * self.args.sigma**2 / self.m
        K_dict = {}
        alpha_dict = {}
        for num_remove in num_remove_list:
            K_list = {}
            alpha_list = {}
            for target_epsilon in epsilon_list:
                K = 1
                epsilon_of_alpha = lambda alpha: self.epsilon_expression(K, sigma, self.eta, C_lsi, alpha, num_remove, self.M, self.m, self.n, self.delta)
                min_epsilon_with_k = minimize_scalar(epsilon_of_alpha, bounds=(1, 10000), method='bounded')
                while min_epsilon_with_k.fun > target_epsilon:
                    K = K + 10
                    epsilon_of_alpha = lambda alpha: self.epsilon_expression(K, sigma, self.eta, C_lsi, alpha, num_remove, self.M, self.m, self.n, self.delta)
                    min_epsilon_with_k = minimize_scalar(epsilon_of_alpha, bounds=(1, 10000), method='bounded')
                K_list[target_epsilon] = K
                alpha_list[target_epsilon] = min_epsilon_with_k.x
                print('num remove:'+str(num_remove)+'target epsilon: '+str(target_epsilon)+'K: '+str(K)+'alpha: '+str(min_epsilon_with_k.x))
            K_dict[num_remove] = K_list
            alpha_dict[num_remove] = alpha_list
        return K_dict, alpha_dict
    def calculate_epsilon0(self, alpha, S, sigma):
        return (4 * alpha * float(S)**2 * float(self.M)**2) / (float(self.m) * float(sigma)**2 * float(self.n)**2)

    def get_mean_performance(self, X, y, step, sigma, w_list, projection, batch_size, batch_idx, len_list = 1, return_w = False, num_trial = 100):
        new_w_list = []
        trial_list = []
        time_list = []
        if w_list is None:
            for trial_idx in tqdm(range(num_trial)):
                w_init, time = self.run_stochastic_gradient_descent(None, X, y, step, sigma, len_list, 
                                                                    projection = projection, batch_size = batch_size, batch_idx = batch_idx)
                time_list.append(time)
                if self.num_class == 2:
                    w_init = np.vstack(w_init)
                else:
                    w_init = np.stack(w_init, axis = 0)
                new_w_list.append(w_init)
                accuracy = self.test_accuracy(w_init)
                trial_list.append(accuracy)
        else:
            for trial_idx in tqdm(range(num_trial)):
                if self.num_class == 2:
                    w = w_list[trial_idx].reshape(-1)
                elif self.num_class == 10:
                    w = w_list[trial_idx].reshape(self.dim_w, -1)
                w = torch.tensor(w)
                new_w, time = self.run_stochastic_gradient_descent(w, X, y, step, sigma, len_list = 1,
                                                                   projection=projection, batch_size=batch_size, batch_idx = batch_idx)
                time_list.append(time)
                if self.num_class == 2:
                    new_w = np.vstack(new_w)
                else:
                    new_w = np.stack(new_w, axis = 0)
                new_w_list.append(new_w)
                accuracy = self.test_accuracy(new_w)
                trial_list.append(accuracy)
        mean_time = np.mean(time_list)

        if return_w:
            new_w_list = np.stack(new_w_list, axis=0)
            return trial_list, mean_time, new_w_list
        else:
            return trial_list, mean_time
        
    def search_burnin(self, sigma_list, burn_in_list, fig_path = '_search_burnin.pdf'):
        acc_dict = {}
        for sigma in sigma_list:
            acc_list = []
            this_w_list = None
            for idx in range(len(burn_in_list)):
                if idx == 0:
                    step = burn_in_list[idx]
                else:
                    step = burn_in_list[idx] - burn_in_list[idx - 1]
                accuracy, _, new_w_list = self.get_mean_performance(self.X_train, self.y_train, step, sigma, this_w_list, return_w = True,
                                                                    projection = self.projection, batch_size = self.batch_size, batch_idx = self.batch_idx)
                this_w_list = new_w_list
                acc_list.append(np.mean(accuracy))
                print(acc_list)
            plt.plot(burn_in_list, acc_list, label='sigma :'+str(sigma))
            acc_dict[sigma] = acc_list
            for i in range(len(burn_in_list)):
                plt.text(burn_in_list[i], acc_list[i], f'{acc_list[i]:.3f}', ha='right', va='bottom')
        plt.legend()
        plt.title(str(self.args.dataset)+'search burn in')
        plt.xlabel('burn in steps')
        plt.ylabel('accuracy')
        plt.savefig(str(self.args.dataset)+fig_path)
        plt.clf()
        return acc_dict
    
    def search_batch(self, burn_in_list, batch_list, fig_path = '_search_batch.pdf'):
        acc_dict = {}
        for batch in batch_list:
            acc_list = []
            this_w_list = None
            for idx in range(len(burn_in_list)):
                if idx == 0:
                    step = burn_in_list[idx]
                else:
                    step = burn_in_list[idx] - burn_in_list[idx - 1]
                accuracy, _, new_w_list = self.get_mean_performance(self.X_train, self.y_train, step, self.args.sigma, this_w_list, return_w = True,
                                                                    projection = self.projection, batch_size = batch, batch_idx = self.batch_idx)
                this_w_list = new_w_list
                acc_list.append(np.mean(accuracy))
                print(acc_list)
            plt.plot(burn_in_list, acc_list, label='batch:'+str(batch))
            acc_dict[batch] = acc_list
            for i in range(len(burn_in_list)):
                plt.text(burn_in_list[i], acc_list[i], f'{acc_list[i]:.3f}', ha='right', va='bottom')
        plt.legend()
        plt.title(str(self.args.dataset)+'search burn in')
        plt.xlabel('burn in steps')
        plt.ylabel('accuracy')
        plt.savefig(str(self.args.dataset)+fig_path)
        plt.clf()
        return acc_dict
                
    def test_accuracy(self, w_list):
        w = torch.tensor(w_list[0])
        if self.num_class == 2:
            pred = self.X_test.mv(w)
            accuracy = pred.gt(0).eq(self.y_test.gt(0)).float().mean()
        elif self.num_class == 10:
            pred = torch.matmul(self.X_test.view(-1, 1, self.dim_w), w.unsqueeze(0))
            _, y_pred = torch.max(pred.squeeze(1), dim = 1)
            y_mask = self.y_test > 0
            y_label = torch.nonzero(y_mask, as_tuple=True)[1]
            accuracy = y_pred.eq(y_label).float().mean()
        return accuracy
    def run_stochastic_gradient_descent(self, init_point, X, y, burn_in, sigma, len_list, projection, batch_size, batch_idx):
        start_time = time.time()
        w_list = stochastic_gradient_descent_algorithm(init_point, self.dim_w, X, y, self.args.lam*self.n, sigma = sigma, 
                                               device = self.device, burn_in = burn_in, 
                                               len_list = len_list, step=self.eta, M = self.M, m = self.m,
                                               projection = projection, batch_size = batch_size, batch_idx = batch_idx,
                                               num_class = self.num_class)
        end_time = time.time()
        return w_list, end_time - start_time

def main():
    parser = argparse.ArgumentParser(description='Training a removal-enabled linear model and testing removal')
    parser.add_argument('--data-dir', type=str, default='./data', help='data directory')
    parser.add_argument('--result-dir', type=str, default='./result', help='directory for saving results')
    parser.add_argument('--dataset', type=str, default='MNIST', help='[MNIST, 2dgaussian, kdgaussian]')
    parser.add_argument('--extractor', type=str, default='raw_feature', help='extractor type')
    parser.add_argument('--lam', type=float, default=1e-7, help='L2 regularization')
    parser.add_argument('--num-removes', type=int, default=1000, help='number of data points to remove')
    parser.add_argument('--num-steps', type=int, default=10000, help='number of optimization steps')
    parser.add_argument('--train-mode', type=str, default='binary', help='train mode [ovr/binary]')
    parser.add_argument('--M', type = float, default = 1, help = 'set M-Lipschitz constant (norm of gradient)')
    parser.add_argument('--projection', type = float, default = 20.0, help = 'set the weight projection radius')
    parser.add_argument('--batch_size', type = int, default = 0, help = 'the batch size')

    parser.add_argument('--gpu', type = int, default = 6, help = 'gpu')
    parser.add_argument('--sigma', type = float, default = 0.01, help = 'the parameter sigma')

    parser.add_argument('--search_burnin', type = int, default = 0, help = 'whether grid search to paint for burn-in')
    parser.add_argument('--search_batch', type = int, default = 0, help = 'paint the batch size utility relation')
    parser.add_argument('--paint_utility_s', type = int, default = 0, help = 'paint the utility - s figure')
    parser.add_argument('--paint_utility_epsilon', type = int, default = 0, help = 'paint utility - epsilon figure')
    parser.add_argument('--paint_unlearning_sigma', type = int, default = 0, help = 'paint unlearning utility - sigma figure')
    parser.add_argument('--how_much_retrain', type = int, default = 0, help = 'supplementary for unlearning sigma')
    args = parser.parse_args()
    print(args)

    runner = Runner(args)
    runner.get_metadata()

    #import pdb; pdb.set_trace()

    runner.train()




if __name__ == '__main__':
    main()