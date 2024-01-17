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

from utils import load_features, generate_gaussian, plot_2dgaussian, plot_w_2dgaussian
from langevin import unadjusted_langevin_algorithm
from density import logistic_density, logistic_potential


class Runner():
    def __init__(self, args):
        self.device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')
        self.args = args
        if args.dataset == 'MNIST':
            self.X_train, self.X_test, self.y_train, self.y_train_onehot, self.y_test = load_features(args)
            self.dim_w = 784
        elif args.dataset == '2dgaussian':
            mean_1 = torch.tensor([-2, -2])
            mean_2 = torch.tensor([2, 2])
            std = torch.tensor([1, 1])
            self.X_train, self.y_train = generate_gaussian(2, 10000, mean_1, mean_2, std)
            self.X_test, self.y_test = generate_gaussian(2, 1000, mean_1, mean_2, std)
            self.dim_w = 2
        elif args.dataset == 'kdgaussian':
            mean_1 = torch.ones(args.gaussian_dim) * -2
            mean_2 = torch.ones(args.gaussian_dim) * 2
            std = torch.ones(args.gaussian_dim)
            self.X_train, self.y_train = generate_gaussian(args.gaussian_dim, 10000, mean_1, mean_2, std)
            self.X_test, self.y_test = generate_gaussian(args.gaussian_dim, 1000, mean_1, mean_2, std)
            self.dim_w = args.gaussian_dim
        # make the norm of x = 1, MNIST naturally satisfys
        self.X_train_norm = self.X_train.norm(dim=1, keepdim=True)
        self.X_train = self.X_train / self.X_train_norm
        self.X_test_norm = self.X_test.norm(dim=1, keepdim=True)
        self.X_test = self.X_test / self.X_test_norm
        self.X_train = self.X_train.to(self.device)
        self.y_train = self.y_train.to(self.device)
    def get_metadata(self):
        # num of training data
        self.n = len(self.X_train)
        print('number training data:'+str(self.n))
        # L-smoothness constant
        X = self.X_train.cpu().numpy()
        self.L = np.max(np.linalg.eigvalsh(X.T @ X / self.n)) / 4 + self.args.lam * self.n
        self.L = 100 * self.L
        print('L smooth constant'+str(self.L))
        # m-strongly convex constant
        self.m = self.args.lam * self.n
        self.m = 100 * self.m
        print('m strongly convex:'+str(self.m))
        # M-Lipschitz constant
        self.M = self.args.M
        print('M lipschitz constant:'+str(self.M))
        # calculate step size
        max_eta = 2 / self.L 
        self.eta = max_eta
        print('step size eta:'+str(self.eta))
        # calculate RDP delta
        self.delta = 1 / self.n
        print('RDP constant delta:'+str(self.delta))
        self.projection = self.args.projection
        print('projection diameter:'+str(self.projection))
        
    def train(self):
        if self.args.paint_utility_s:
            num_remove_list = [1, 10, 50, 100, 500, 1000]
            batch_list = [1, 2048, 4096, self.n]
            #burn_in_list = [100000, 40000, 20000, 10000]
            burn_in_list = [100,100,100,1000]
            
            for idx, batch in enumerate(batch_list):
                scratch_acc_list = []
                unlearn_acc_list = []
                if batch == self.n:
                    avg_accuracy_scratch_D, mean_time, w_list = self.get_mean_performance(self.X_train, self.y_train, burn_in_list[idx], 
                                                                                      self.args.sigma, None, self.projection, None, len_list = 1, return_w = True)
                else:
                    avg_accuracy_scratch_D, mean_time, w_list = self.get_mean_performance(self.X_train, self.y_train, burn_in_list[idx], 
                                                                                      self.args.sigma, None, self.projection, batch, len_list = 1, return_w = True)
                np.save('./result/SGD/paint_utility_s/learn_scratch_w_'+str(batch)+'.npy', w_list)
                scratch_acc_list.append(avg_accuracy_scratch_D)
                unlearn_acc_list.append(avg_accuracy_scratch_D)
                # calculate K
                epsilon_list = [1] # set epsilon = 1
                old_k, alpha_dict = self.search_finetune_step(epsilon_list, num_remove_list)
                K_dict = self.search_sgd_step(epsilon_list, alpha_dict, batch)
                K_list = []
                for num_remove in num_remove_list:
                    X_train_removed, y_train_removed = self.get_removed_data(num_remove)
                    if batch == self.n:
                        avg_accuracy_scratch_Dnew, mean_time = self.get_mean_performance(X_train_removed, y_train_removed, burn_in_list[idx], 
                                                                                     self.args.sigma, None, self.projection, None)
                    else:
                        avg_accuracy_scratch_Dnew, mean_time = self.get_mean_performance(X_train_removed, y_train_removed, burn_in_list[idx], 
                                                                                     self.args.sigma, None, self.projection, batch)
                    scratch_acc_list.append(avg_accuracy_scratch_Dnew)
                    if batch == self.n:
                        avg_accuracy_finetune, mean_time = self.get_mean_performance(X_train_removed, y_train_removed, K_dict[num_remove][1], 
                                                                                     self.args.sigma, w_list, self.projection, None)
                    else:
                        avg_accuracy_finetune, mean_time = self.get_mean_performance(X_train_removed, y_train_removed, K_dict[num_remove][1], 
                                                                                     self.args.sigma, w_list, self.projection, batch)
                    unlearn_acc_list.append(avg_accuracy_finetune)
                    K_list.append(K_dict[num_remove][1])
                x_list = [0] + num_remove_list
                np.save('./result/SGD/paint_utility_s/learn_scratch_acc_'+str(batch)+'.npy', scratch_acc_list)
                np.save('./result/SGD/paint_utility_s/unlearn_scratch_acc_'+str(batch)+'.npy', unlearn_acc_list)
                np.save('./result/SGD/paint_utility_s/k_list_'+str(batch)+'.npy', K_list)
                plt.plot(x_list, scratch_acc_list, label='learn from scratch, b = '+str(batch), marker = 'o')
                plt.plot(x_list, unlearn_acc_list, label='unlearn, b = '+str(batch), marker = 'o')
                plt.legend()
                for i, k in enumerate(K_list):
                    plt.text(x_list[i+1], unlearn_acc_list[i+1], f'K = {k}', fontsize=8)
            plt.xlabel('# removed data')
            plt.ylabel('test accuracy')
            plt.savefig('./result/SGD/paint_utility_s/paint_utility_s.pdf')
            plt.clf()
        elif self.args.paint_utility_epsilon:
            epsilon_list = [0.1, 0.5, 1, 2, 5]
            num_remove_list = [1000]
            batch_list = [1, 4096, self.n]
            #burn_in_list = [100000, 20000, 10000]
            burn_in_list = [10, 10, 10]
            for idx, batch in enumerate(batch_list):
                unlearn_acc_list = []
                if batch == self.n:
                    avg_accuracy_scratch_D, mean_time, w_list = self.get_mean_performance(self.X_train, self.y_train, burn_in_list[idx], 
                                                                                          self.args.sigma, None, self.projection, None, len_list = 1, return_w = True)
                else:
                    avg_accuracy_scratch_D, mean_time, w_list = self.get_mean_performance(self.X_train, self.y_train, burn_in_list[idx], 
                                                                                          self.args.sigma, None, self.projection, batch, len_list = 1, return_w = True)
                np.save('./result/SGD/paint_utility_epsilon/w_from_scratch'+str(batch)+'.npy', w_list)
                unlearn_acc_list.append(avg_accuracy_scratch_D)
                # calculate K
                _, alpha_dict = self.search_finetune_step(epsilon_list, num_remove_list)
                K_dict = self.search_sgd_step(epsilon_list, alpha_dict, batch)
                np.save('./result/SGD/paint_utility_epsilon/K_list'+str(batch)+'.npy', K_dict)
                K_list = []
                for epsilon in epsilon_list:
                    X_train_removed, y_train_removed = self.get_removed_data(num_remove_list[0])
                    if batch == self.n:
                        avg_accuracy_finetune, mean_time = self.get_mean_performance(X_train_removed, y_train_removed, K_dict[num_remove_list[0]][epsilon], 
                                                                                     self.args.sigma, w_list, self.projection, None)
                    else:
                        avg_accuracy_finetune, mean_time = self.get_mean_performance(X_train_removed, y_train_removed, K_dict[num_remove_list[0]][epsilon], 
                                                                                     self.args.sigma, w_list, self.projection, batch)
                    unlearn_acc_list.append(avg_accuracy_finetune)
                    K_list.append(K_dict[num_remove_list[0]][epsilon])
                np.save('./result/SGD/paint_utility_epsilon/unlearn_acc_list'+str(batch)+'.npy', unlearn_acc_list)
                x_list = [0] + epsilon_list
                plt.plot(x_list, unlearn_acc_list, label='unlearn, b = '+str(batch), marker = 'o')
                plt.legend()
                for i, k in enumerate(K_list):
                    plt.text(x_list[i+1], unlearn_acc_list[i+1], f'K = {k}', fontsize=8)
            plt.xlabel('epsilon')
            plt.ylabel('test accuracy')
            plt.savefig('./result/SGD/paint_utility_epsilon/paint_utility_epsilon.pdf')
            plt.clf()
        elif self.args.paint_unlearning_sigma:
            num_remove_list = [1000]
            epsilon_list = [1]
            sigma_list = [0.05, 0.1, 0.2, 0.5, 1]
            _, alpha_dict = self.search_finetune_step(epsilon_list, num_remove_list)
            batch_list = [1, 4096, self.n]
            #burn_in_list = [100000, 20000, 10000]
            burn_in_list = [10, 20, 10]

            for idx, batch in enumerate(batch_list):
                K_dict = self.search_sgd_step(epsilon_list, alpha_dict, batch)
                np.save('./result/SGD/paint_unlearning_sigma/K_dict.npy', K_dict)
                alpha = alpha_dict[num_remove_list[0]][epsilon_list[0]]
                
                scratch_acc_list = []
                scratch_unlearn_list = []
                finetune_unlearn_list = []
                X_train_removed, y_train_removed = self.get_removed_data(num_remove_list[0])
                for sigma in sigma_list:
                    if batch == self.n:
                        avg_accuracy_scratch_D, mean_time, w_list = self.get_mean_performance(self.X_train, self.y_train, burn_in_list[idx], sigma, 
                                                                                              None, self.projection, None, len_list = 1, return_w = True)
                    else:
                        avg_accuracy_scratch_D, mean_time, w_list = self.get_mean_performance(self.X_train, self.y_train, burn_in_list[idx], sigma, 
                                                                                              None, self.projection, batch, len_list = 1, return_w = True)
                    np.save('./result/SGD/paint_unlearning_sigma/'+str(sigma)+'_learn_scratch_w'+str(batch)+'.npy', w_list)
                    scratch_acc_list.append(avg_accuracy_scratch_D)
                    if batch == self.n:
                        avg_accuracy_scratch_Dnew, mean_time, unlearn_w_list = self.get_mean_performance(X_train_removed, y_train_removed, burn_in_list[idx], sigma, 
                                                                                                         None, self.projection, None, return_w=True)
                    else:
                        avg_accuracy_scratch_Dnew, mean_time, unlearn_w_list = self.get_mean_performance(X_train_removed, y_train_removed, burn_in_list[idx], sigma, 
                                                                                                         None, self.projection, batch, return_w=True)
                    np.save('./result/SGD/paint_unlearning_sigma/'+str(sigma)+'_unlearn_scratch_w'+str(batch)+'.npy', unlearn_w_list)
                    scratch_unlearn_list.append(avg_accuracy_scratch_Dnew)
                    if batch == self.n:
                        avg_accuracy_finetune, mean_time = self.get_mean_performance(X_train_removed, y_train_removed, K_dict[num_remove_list[0]][1], sigma, 
                                                                                     w_list, self.projection, None)
                    else:
                        avg_accuracy_finetune, mean_time = self.get_mean_performance(X_train_removed, y_train_removed, K_dict[num_remove_list[0]][1], sigma, 
                                                                                     w_list, self.projection, batch)
                    finetune_unlearn_list.append(avg_accuracy_finetune)
                plt.plot(sigma_list, scratch_acc_list, label = 'learn D, b='+str(batch), marker = 'o')
                plt.plot(sigma_list, scratch_unlearn_list, label = 'learn D\', b = '+str(batch), marker = 'o')
                plt.plot(sigma_list, finetune_unlearn_list, label = 'unlearn D\', b = '+str(batch), marker = 'o')
                np.save('./result/SGD/paint_unlearning_sigma/learn_scratch_acc'+str(batch)+'.npy', scratch_acc_list)
                np.save('./result/SGD/paint_unlearning_sigma/unlearn_scratch_acc'+str(batch)+'.npy', scratch_unlearn_list)
                np.save('./result/SGD/paint_unlearning_sigma/unlearn_finetune_acc'+str(batch)+'.npy', finetune_unlearn_list)
            plt.xlabel(r'$\sigma$')
            plt.ylabel('test accuracy')
            plt.legend()
            plt.legend()
            plt.savefig('./result/SGD/paint_unlearning_sigma/paint_unlearning_sigma.pdf')
            plt.clf()
        elif self.args.compare_k == 1:
            epsilon_list = [0.1, 0.5, 1, 2, 5]
            num_remove_list = [1000]
            b_list = [2048, 4096, self.n]
            K_dict, alpha_dict = self.search_finetune_step(epsilon_list, num_remove_list)
            # alpha dict: key = num_remove, then has a dict with key = epsilon, item = alpha
            # K dict: key = num_remove, then has a dict with key = epsilon, item = K
            for key in K_dict.keys():
                tmp_K_dict = K_dict[key]
                K_list = []
                for target_epsilon in epsilon_list:
                    K_list.append(tmp_K_dict[target_epsilon])
                #plt.plot(epsilon_list, np.array(K_list)*self.n, label='K (SGD)', marker = 'o')
            for b in b_list:
                K_sgd_dict = self.search_sgd_step(epsilon_list, alpha_dict, b)
                for key in K_dict.keys():
                    tmp_K_sgd_dict = K_sgd_dict[key]
                    K_sgd_list = []
                    for target_epsilon in epsilon_list:
                        K_sgd_list.append(tmp_K_sgd_dict[target_epsilon])
                        print(K_sgd_list)
                    plt.plot(epsilon_list, np.array(K_sgd_list), label='K (sgd) b='+str(b), marker = 'o')
            
            plt.legend()
            plt.xlabel(r'target $\epsilon$')
            plt.ylabel(r'required step K')
            plt.savefig('./compare_stepK.pdf')
            plt.clf()

        elif self.args.find_best_batch == 1:
            epsilon_list = [0.1, 0.5, 1, 2, 5]
            num_remove_list = [1000]
            b_list = list(range(1, self.n, 10))
            K_dict, alpha_dict = self.search_finetune_step(epsilon_list, num_remove_list)
            for epsilon in tqdm(epsilon_list):
                tmp_K_sgd_list = []
                for b in tqdm(b_list):
                    K_sgd_dict = self.search_sgd_step([epsilon], alpha_dict, b)
                    tmp_K_sgd_list.append(K_sgd_dict[num_remove_list[0]][epsilon]*b)
                plt.plot(b_list, np.array(tmp_K_sgd_list), label=r'K ($\epsilon$)='+str(epsilon))
            plt.legend()
            plt.xlabel(r'batch size b')
            plt.ylabel(r'required step K * b')
            plt.savefig('./find_best_b.pdf')
            plt.clf()


        else:
            print('check!')
    
    def get_z(self, num_remove, b):
        # get the Z in SGD
        # b is the batch size
        #C = ((b - num_remove) / b) * (1 - self.eta * self.L * self.m / (self.L + self.m)) + num_remove / b
        C = 1 - self.eta * self.m
        T = sp.symbols('T')
        fT = ((1 - C**T) / (1 - C)) * ((2 * num_remove * self.eta * self.M) / b)
        part_1 = float(sp.limit(fT, T, sp.oo))
        part_2 = self.projection
        return min(part_1, part_2)
    
    def search_sgd_step(self, epsilon_list, alpha_dict, b):
        K_sgd_dict = {}
        for key in alpha_dict.keys():
            # here key = num_remove = S
            tmp_alpha_dict = alpha_dict[key]
            tmp_K_dict = {}
            for target_epsilon in epsilon_list:
                target_alpha = tmp_alpha_dict[target_epsilon]
                if key < b:
                    Z = self.get_z(key, b)
                else:
                    Z = self.projection
                #c = 1 - (self.eta * (self.L * self.m) / (self.L + self.m))
                c = 1 - self.eta * self.m
                K = sp.symbols('K', integer = True)
                part_1 = ((target_alpha * Z**2) / (2 * self.eta * self.args.sigma**2)) * ((c**2 - 1))
                inequality1 =  part_1 / (1 - (1 / c**(2 * K))) - target_epsilon < 0
                solutions1 = sp.solve(inequality1, K)
                num_list = re.findall(r"[-+]?\d*\.\d+|\d+", str(solutions1))
                num_list = [float(num) for num in num_list if float(num) > 0]
                solution = math.ceil(num_list[0])
                tmp_K_dict[target_epsilon] = solution
            K_sgd_dict[key] = tmp_K_dict
        return K_sgd_dict

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
        #part_1 = math.exp(- (2 * float(K) * float(sigma) **2 * float(eta)) / (alpha * float(C_lsi)))
        part_1 = math.exp(- (float(K) * m * float(eta)) / (alpha))
        part_2 = (4 * alpha * float(S)**2 * float(M)**2) / (float(m) * float(sigma)**2 * float(n)**2)
        part_3 = (math.log(1 / float(delta))) / (alpha - 1)
        epsilon = part_1 * part_2 + part_3
        return epsilon
    
    def search_finetune_step(self, epsilon_list, num_remove_list):
        C_lsi = 2 * self.args.sigma**2 / self.m
        K_dict = {}
        alpha_dict = {}
        for num_remove in num_remove_list:
            K_list = {}
            alpha_list = {}
            for target_epsilon in epsilon_list:
                K = 1
                epsilon_of_alpha = lambda alpha: self.epsilon_expression(K, self.args.sigma, self.eta, C_lsi, alpha, num_remove, self.M, self.m, self.n, self.delta)
                min_epsilon_with_k = minimize_scalar(epsilon_of_alpha, bounds=(1, 10000), method='bounded')
                while min_epsilon_with_k.fun > target_epsilon:
                    K = K + 10
                    epsilon_of_alpha = lambda alpha: self.epsilon_expression(K, self.args.sigma, self.eta, C_lsi, alpha, num_remove, self.M, self.m, self.n, self.delta)
                    min_epsilon_with_k = minimize_scalar(epsilon_of_alpha, bounds=(1, 10000), method='bounded')
                K_list[target_epsilon] = K
                alpha_list[target_epsilon] = min_epsilon_with_k.x
                print('num remove:'+str(num_remove)+'target epsilon: '+str(target_epsilon)+'K: '+str(K)+'alpha: '+str(min_epsilon_with_k.x))
            K_dict[num_remove] = K_list
            alpha_dict[num_remove] = alpha_list
        return K_dict, alpha_dict
    def calculate_epsilon0(self, alpha, S, sigma):
        return (4 * alpha * float(S)**2 * float(self.M)**2) / (float(self.m) * float(sigma)**2 * float(self.n)**2)

    def get_mean_performance(self, X, y, step, sigma, w_list, projection = None, batch = None, len_list = 1, return_w = False, num_trial = 100):
        new_w_list = []
        trial_list = []
        time_list = []
        if w_list is None:
            for trial_idx in tqdm(range(num_trial)):
                w_init, time = self.run_unadjusted_langvin(None, X, y, step, sigma, len_list, projection, batch)
                time_list.append(time)
                w_init = np.vstack(w_init)
                new_w_list.append(w_init)
                accuracy = self.test_accuracy(w_init)
                trial_list.append(accuracy)
        else:
            for trial_idx in tqdm(range(num_trial)):
                w = w_list[trial_idx].reshape(-1)
                w = torch.tensor(w)
                new_w, time = self.run_unadjusted_langvin(w, X, y, step, sigma, len_list = 1, projection = projection, batch_size = batch)
                time_list.append(time)
                new_w = np.vstack(new_w)
                new_w_list.append(new_w)
                accuracy = self.test_accuracy(new_w)
                trial_list.append(accuracy)
        avg_accuracy = np.mean(trial_list)
        mean_time = np.mean(time_list)

        if return_w:
            new_w_list = np.stack(new_w_list, axis=0)
            return avg_accuracy, mean_time, new_w_list
        else:
            return avg_accuracy, mean_time
    
    def test_accuracy(self, w_list):
        w = torch.tensor(w_list[0])
        # test accuracy (before removal)
        pred = self.X_test.mv(w)
        accuracy = pred.gt(0).eq(self.y_test.gt(0)).float().mean()
        return accuracy
    def run_unadjusted_langvin(self, init_point, X, y, burn_in, sigma, len_list, projection = None, batch_size = None):
        start_time = time.time()
        w_list = unadjusted_langevin_algorithm(init_point, self.dim_w, X, y, self.args.lam, sigma = sigma, 
                                               device = self.device, potential = logistic_potential, burn_in = burn_in, 
                                               len_list = len_list, step=self.eta, M = self.M, projection = projection, batch_size = batch_size)
        end_time = time.time()
        return w_list, end_time - start_time

def main():
    parser = argparse.ArgumentParser(description='Training a removal-enabled linear model and testing removal')
    parser.add_argument('--data-dir', type=str, default='./data', help='data directory')
    parser.add_argument('--result-dir', type=str, default='./result', help='directory for saving results')
    parser.add_argument('--dataset', type=str, default='MNIST', help='[MNIST, 2dgaussian, kdgaussian]')
    parser.add_argument('--extractor', type=str, default='raw_feature', help='extractor type')
    parser.add_argument('--lam', type=float, default=1e-6, help='L2 regularization')
    parser.add_argument('--num-removes', type=int, default=1000, help='number of data points to remove')
    parser.add_argument('--num-steps', type=int, default=10000, help='number of optimization steps')
    parser.add_argument('--train-mode', type=str, default='binary', help='train mode [ovr/binary]')
    parser.add_argument('--M', type = float, default = 1, help = 'set M-Lipschitz constant (norm of gradient)')
    parser.add_argument('--projection', type = float, default = 1, help = 'set the projected ball diameter D')

    parser.add_argument('--gpu', type = int, default = 6, help = 'gpu')
    parser.add_argument('--sigma', type = float, default = 0.1, help = 'the parameter sigma')
    parser.add_argument('--burn_in', type = int, default = 35000, help = 'burn in step number of SGD')
    parser.add_argument('--gaussian_dim', type = int, default = 10, help = 'dimension of gaussian task')
    parser.add_argument('--len_list', type = int, default = 10000, help = 'length of w to paint in 2D gaussian')
    parser.add_argument('--finetune_step', type = int, default = 50, help = 'steps to finetune on the new removed data')


    parser.add_argument('--search_burnin', type = int, default = 0, help = 'whether grid search to paint for burn-in')
    parser.add_argument('--search_finetune', type = int, default = 0, help = 'whether to grid search finetune')
    parser.add_argument('--search_burnin_newdata', type = int, default = 0, help = 'search burn in on new data')
    parser.add_argument('--paint_utility_s', type = int, default = 0, help = 'paint the utility - s figure')
    parser.add_argument('--paint_utility_epsilon', type = int, default = 0, help = 'paint utility - epsilon figure')
    parser.add_argument('--paint_unlearning_sigma', type = int, default = 0, help = 'paint unlearning utility - sigma figure')
    parser.add_argument('--compare_k', type = int, default = 0, help = 'calculae the K step required by our bound and baseline bound')
    parser.add_argument('--find_best_batch', type = int, default = 0, help = 'find the best batch per gradient for sgd')
    args = parser.parse_args()
    print(args)

    runner = Runner(args)
    runner.get_metadata()
    runner.train()


if __name__ == '__main__':
    main()