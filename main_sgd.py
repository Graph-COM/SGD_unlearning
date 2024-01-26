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
            
            self.dim_w = 784
            if args.dataset == 'MNIST':
                self.num_class = 2
                self.X_train = self.X_train[:11264]
                self.y_train = self.y_train[:11264]
            else:
                self.num_class = 10
                self.X_train = self.X_train[:56320]
                self.y_train = self.y_train[:56320]
        elif args.dataset == 'CIFAR10' or args.dataset == 'CIFAR10_multiclass':
            self.X_train, self.X_test, self.y_train, self.y_train_onehot, self.y_test = load_features(args)
            self.X_train = self.X_train[:9728]
            self.y_train = self.y_train[:9728]
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
        self.eta = 1 / self.L
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
        batch_idx_list = []
        for trial_idx in range(100):
            np.random.shuffle(batch_idx)
            batch_idx_list.append(batch_idx.copy())
        self.batch_idx = batch_idx_list
        print('have shuffled batch idx')
        
    def train(self):
        if self.args.search_burnin:
            # this is for full-batch
            if self.args.dataset == 'MNIST':
                sigma_list = [0.03]
                burn_in_list = [1, 10, 20, 50, 100, 150, 200, 300, 500, 750, 1000]
            elif self.args.dataset == 'MNIST_multiclass':
                sigma_list = [0.005, 0.01]
                burn_in_list = [1, 10, 20, 50, 100, 150, 200]
            elif self.args.dataset == 'CIFAR10':
                sigma_list = [0.03]
                burn_in_list = [1, 10, 20, 50, 100, 150, 200, 300, 500, 750, 1000]
            _ = self.search_burnin(sigma_list, burn_in_list)
        elif self.args.search_batch:
            batch_list = [512]
            burn_in_list = [1, 2, 5, 10, 20, 50, 100, 200, 300, 500]
            _ = self.search_batch(burn_in_list, batch_list)
        elif self.args.compare_baseline:
            # compare with the baseline (remove 1 data see sigma and utility)
            epsilon_list = [0.05, 0.1, 0.5, 1, 2, 5]
            batch_list = [128, 256, 0]
            burn_in_list = [100, 150, 1000]
            create_nested_folder('./result/SGD/'+str(self.args.dataset)+'/baseline/')
            X_train_removed, y_train_removed = self.get_removed_data(1)
            target_k_list = [1]
            for batch_size, burn_in in zip(batch_list, burn_in_list):
                print('working on batch:'+str(batch_size))
                # for each type of batch size
                for target_k in target_k_list:
                    # for each target k
                    sigma_list = []
                    for target_epsilon in epsilon_list:
                        sigma_list.append(self.search_alpha(target_k, target_epsilon, batch_size))
                    print('batch: '+str(batch_size)+'target k:'+str(target_k) + ' sigma: '+str(sigma_list))
                # if it's none, then just take this value
                sigma_list = [x if x is not None else 7.450581596923812e-9 for x in sigma_list]
                # know the required k, and epsilon, sigma
                for epsilon, sigma in zip(epsilon_list, sigma_list):
                    print('working on epsilon:'+str(epsilon))
                    create_nested_folder('./result/SGD/'+str(self.args.dataset)+'/baseline/'+str(target_k)+'/')
                    sgd_learn_scratch_acc, mean_time, sgd_w_list = self.get_mean_performance(self.X_train, self.y_train, burn_in, sigma, None,
                                                                                             self.projection, batch_size, self.batch_idx, len_list = 1, return_w = True)
                    print('SGD learn scratch acc: ' + str(np.mean(sgd_learn_scratch_acc)))
                    print('SGD learn scratch acc std: ' + str(np.std(sgd_learn_scratch_acc)))
                    np.save('./result/SGD/'+str(self.args.dataset)+'/baseline/'+str(target_k)+'/sgd_acc_learn_scratch_b'+str(batch_size)+'_eps'+str(epsilon)+'.npy', sgd_learn_scratch_acc)
                    sgd_unlearn_scratch_acc, mean_time = self.get_mean_performance(X_train_removed, y_train_removed, burn_in, sigma, None, 
                                                                                   self.projection, batch_size, self.batch_idx, len_list = 1)
                    print('SGD unlearn scratch acc: ' + str(np.mean(sgd_unlearn_scratch_acc)))
                    print('SGD unlearn scratch acc std: ' + str(np.std(sgd_unlearn_scratch_acc)))
                    np.save('./result/SGD/'+str(self.args.dataset)+'/baseline/'+str(target_k)+'/sgd_acc_unlearn_scratch_b'+str(batch_size)+'_eps'+str(epsilon)+'.npy', sgd_unlearn_scratch_acc)
                    sgd_unlearn_finetune_acc, mean_time = self.get_mean_performance(X_train_removed, y_train_removed, target_k_list[0], sigma, sgd_w_list,
                                                                                    self.projection, batch_size, self.batch_idx, len_list = 1)
                    print('SGD unlearn finetune acc: ' + str(np.mean(sgd_unlearn_finetune_acc)))
                    print('SGD unlearn finetune acc std: ' + str(np.std(sgd_unlearn_finetune_acc)))
                    np.save('./result/SGD/'+str(self.args.dataset)+'/baseline/'+str(target_k)+'/sgd_acc_unlearn_finetune_b'+str(batch_size)+'_eps'+str(epsilon)+'.npy', sgd_unlearn_finetune_acc)
        elif self.args.compare_baseline_nonconvergent:
            # compare with the baseline nonconvergent (remove 1 data see sigma and utility)
            epsilon_list = [0.05, 0.1, 0.5, 1, 2, 5]
            batch_list = [128, 0]
            burn_in_list = [100, 3000]
            create_nested_folder('./result/SGD/'+str(self.args.dataset)+'/baseline_nonconvergent/')
            X_train_removed, y_train_removed = self.get_removed_data(1)
            target_k_list = [1]
            for batch_size, burn_in in zip(batch_list, burn_in_list):
                print('working on batch:'+str(batch_size))
                # for each type of batch size
                for target_k in target_k_list:
                    # for each target k
                    sigma_list = []
                    for target_epsilon in epsilon_list:
                        sigma_list.append(self.search_alpha_nonconvergent(target_k, target_epsilon, batch_size, burn_in, self.projection, 100))
                    print('batch: '+str(batch_size)+'target k:'+str(target_k) + ' sigma: '+str(sigma_list))
                # if it's none, then just take this value
                sigma_list = [x if x is not None else 7.450581596923812e-9 for x in sigma_list]
                # know the required k, and epsilon, sigma
                for epsilon, sigma in zip(epsilon_list, sigma_list):
                    print('working on epsilon:'+str(epsilon))
                    create_nested_folder('./result/SGD/'+str(self.args.dataset)+'/baseline_nonconvergent/'+str(target_k)+'/')
                    sgd_learn_scratch_acc, mean_time, sgd_w_list = self.get_mean_performance(self.X_train, self.y_train, burn_in, sigma, None,
                                                                                                self.projection, batch_size, self.batch_idx, len_list = 1, return_w = True)
                    print('SGD learn scratch acc: ' + str(np.mean(sgd_learn_scratch_acc)))
                    print('SGD learn scratch acc std: ' + str(np.std(sgd_learn_scratch_acc)))
                    np.save('./result/SGD/'+str(self.args.dataset)+'/baseline_nonconvergent/'+str(target_k)+'/sgd_acc_learn_scratch_b'+str(batch_size)+'_eps'+str(epsilon)+'.npy', sgd_learn_scratch_acc)
                    sgd_unlearn_scratch_acc, mean_time = self.get_mean_performance(X_train_removed, y_train_removed, burn_in, sigma, None, 
                                                                                    self.projection, batch_size, self.batch_idx, len_list = 1)
                    print('SGD unlearn scratch acc: ' + str(np.mean(sgd_unlearn_scratch_acc)))
                    print('SGD unlearn scratch acc std: ' + str(np.std(sgd_unlearn_scratch_acc)))
                    np.save('./result/SGD/'+str(self.args.dataset)+'/baseline_nonconvergent/'+str(target_k)+'/sgd_acc_unlearn_scratch_b'+str(batch_size)+'_eps'+str(epsilon)+'.npy', sgd_unlearn_scratch_acc)
                    sgd_unlearn_finetune_acc, mean_time = self.get_mean_performance(X_train_removed, y_train_removed, target_k_list[0], sigma, sgd_w_list,
                                                                                    self.projection, batch_size, self.batch_idx, len_list = 1)
                    print('SGD unlearn finetune acc: ' + str(np.mean(sgd_unlearn_finetune_acc)))
                    print('SGD unlearn finetune acc std: ' + str(np.std(sgd_unlearn_finetune_acc)))
                    np.save('./result/SGD/'+str(self.args.dataset)+'/baseline_nonconvergent/'+str(target_k)+'/sgd_acc_unlearn_finetune_b'+str(batch_size)+'_eps'+str(epsilon)+'.npy', sgd_unlearn_finetune_acc)
        elif self.args.sequential:
            num_remove_list = [100]
            num_step = num_remove_list[0]
            target_epsilon = 1
            create_nested_folder('./result/SGD/'+str(self.args.dataset)+'/sequential/')
            sigma = 0.03
            #batch_list = [128, 256, 512, 0]
            #burn_in_list = [100, 150, 200, 1000]
            batch_list = [32, 64]
            burn_in_list = [50, 75]
            for batch_size, burn_in in zip(batch_list, burn_in_list):
                print('working on batch size '+str(batch_size))
                self.k_list = np.zeros(num_step+1).astype(int)
                self.ZB_list = np.zeros(num_step+1)
                self.ZB_list[0] = self.Z_B_loose(batch_size)
                self.ZB_list[1] = self.Z_B_loose(batch_size)
                k_1, _ = self.compute_k_loose(sigma, target_epsilon, batch_size)
                self.k_list[1] = k_1
                for step in tqdm(range(2, num_step + 1)):
                    self.k_list[step] = 1
                    self.ZB_list[step] = self.Z_B_sequential(step, batch_size, self.ZB_list[step - 1])
                    epsilon_of_step = lambda alpha: self.epsilon_with_alpha_z(sigma, alpha, self.k_list[step], batch_size, self.ZB_list[step]) + (math.log(1 / float(self.delta))) / (alpha - 1)
                    min_epsilon_step = minimize_scalar(epsilon_of_step, bounds=(1, 100000), method='bounded')
                    while min_epsilon_step.fun > target_epsilon:
                        self.k_list[step] = self.k_list[step] + 1
                        epsilon_of_step = lambda alpha: self.epsilon_with_alpha_z(sigma, alpha, self.k_list[step], batch_size, self.ZB_list[step]) + (math.log(1 / float(self.delta))) / (alpha - 1)
                        min_epsilon_step = minimize_scalar(epsilon_of_step, bounds=(1, 100000), method='bounded')
                print('batch size: '+str(batch_size)+' k list: '+str(self.k_list)+' zb list: '+str(self.ZB_list))
                np.save('./result/SGD/'+str(self.args.dataset)+'/sequential/'+'k_list_b'+str(batch_size)+'.npy', self.k_list)

                # see utility
                sgd_learn_scratch_acc, mean_time, sgd_w_list = self.get_mean_performance(self.X_train, self.y_train, burn_in, sigma, None,
                                                                                        self.projection, batch_size, self.batch_idx, len_list = 1, return_w = True)
                print('SGD learn scratch acc: ' + str(np.mean(sgd_learn_scratch_acc)))
                print('SGD learn scratch acc std: ' + str(np.std(sgd_learn_scratch_acc)))
                np.save('./result/SGD/'+str(self.args.dataset)+'/sequential/sgd_acc_learn_scratch_b'+str(batch_size)+'.npy', sgd_learn_scratch_acc)
                for sgd_step in range(len(self.k_list)-1):
                    sgd_k = self.k_list[sgd_step + 1]
                    X_train_removed, y_train_removed = self.get_removed_data(int(sgd_step+1))
                    sgd_unlearn_finetune_acc, mean_time, sgd_w_list = self.get_mean_performance(X_train_removed, y_train_removed, sgd_k, sigma, sgd_w_list, 
                                                                                                self.projection, batch_size, self.batch_idx, len_list = 1, return_w = True)
                    print('SGD unlearn finetune acc: ' + str(np.mean(sgd_unlearn_finetune_acc)))
                    print('SGD unlearn finetune acc std: ' + str(np.std(sgd_unlearn_finetune_acc)))
                    np.save('./result/SGD/'+str(self.args.dataset)+'/sequential/sgd_acc_finetune_b'+str(batch_size)+'_step'+str(sgd_step+1)+'.npy', sgd_unlearn_finetune_acc)
            import pdb; pdb.set_trace()

        elif self.args.paint_unlearning_sigma:
            num_remove_list = [100]
            num_step = num_remove_list[0]
            target_epsilon = 1
            #sigma_list = [0.05, 0.1, 0.2, 0.5, 1]
            sigma_list = [0.01]
            batch_list = [32, 128, 512, 0]
            #burn_in_list = [100, 150, 200, 1000]
            batch_list = [32, 64]
            burn_in_list = [50, 75]
            create_nested_folder('./result/SGD/'+str(self.args.dataset)+'/paint_unlearning_sigma/')
            for batch_size, burn_in in zip(batch_list, burn_in_list):
                print('working on batch size '+str(batch_size))
                for sigma in sigma_list:
                    self.k_list = np.zeros(num_step+1).astype(int)
                    self.ZB_list = np.zeros(num_step+1)
                    self.ZB_list[0] = self.Z_B_loose(batch_size)
                    self.ZB_list[1] = self.Z_B_loose(batch_size)
                    k_1, _ = self.compute_k_loose(sigma, target_epsilon, batch_size)
                    self.k_list[1] = k_1
                    for step in tqdm(range(2, num_step + 1)):
                        self.k_list[step] = 1
                        self.ZB_list[step] = self.Z_B_sequential(step, batch_size, self.ZB_list[step - 1])
                        epsilon_of_step = lambda alpha: self.epsilon_with_alpha_z(sigma, alpha, self.k_list[step], batch_size, self.ZB_list[step]) + (math.log(1 / float(self.delta))) / (alpha - 1)
                        min_epsilon_step = minimize_scalar(epsilon_of_step, bounds=(1, 100000), method='bounded')
                        while min_epsilon_step.fun > target_epsilon:
                            self.k_list[step] = self.k_list[step] + 1
                            epsilon_of_step = lambda alpha: self.epsilon_with_alpha_z(sigma, alpha, self.k_list[step], batch_size, self.ZB_list[step]) + (math.log(1 / float(self.delta))) / (alpha - 1)
                            min_epsilon_step = minimize_scalar(epsilon_of_step, bounds=(1, 100000), method='bounded')
                    print('batch size: '+str(batch_size)+'sigma: '+str(sigma)+' k list: '+str(self.k_list)+' zb list: '+str(self.ZB_list))
                    np.save('./result/SGD/'+str(self.args.dataset)+'/paint_unlearning_sigma/'+'k_list_b'+str(batch_size)+'_sigma'+str(sigma)+'.npy', self.k_list)
                    # see the utility
                    sgd_learn_scratch_acc, mean_time, sgd_w_list = self.get_mean_performance(self.X_train, self.y_train, burn_in, sigma, None,
                                                                                        self.projection, batch_size, self.batch_idx, len_list = 1, return_w = True)
                    print('SGD learn scratch acc: ' + str(np.mean(sgd_learn_scratch_acc)))
                    print('SGD learn scratch acc std: ' + str(np.std(sgd_learn_scratch_acc)))
                    np.save('./result/SGD/'+str(self.args.dataset)+'/paint_unlearning_sigma/sgd_acc_learn_scratch_b'+str(batch_size)+'_sigma'+str(sigma)+'.npy', sgd_learn_scratch_acc)

                    total_remove_x, total_remove_y = self.get_removed_data(num_remove_list[0])
                    accuracy_scratch_Dnew, mean_time, unlearn_w_list = self.get_mean_performance(total_remove_x, total_remove_y, burn_in, sigma, None,
                                                                                                self.projection, batch_size, self.batch_idx, return_w=True)
                    np.save('./result/SGD/'+str(self.args.dataset)+'/paint_unlearning_sigma/sgd_acc_unlearn_scratch_b'+str(batch_size)+'_sigma'+str(sigma)+'.npy', accuracy_scratch_Dnew)
                    for sgd_step in range(len(self.k_list)-1):
                        sgd_k = self.k_list[sgd_step+1]
                        X_train_removed, y_train_removed = self.get_removed_data(int(sgd_step+1))
                        sgd_unlearn_finetune_acc, mean_time, sgd_w_list = self.get_mean_performance(X_train_removed, y_train_removed, sgd_k, sigma, sgd_w_list, 
                                                                                                    self.projection, batch_size, self.batch_idx, len_list = 1, return_w = True)
                        print('SGD unlearn finetune acc: ' + str(np.mean(sgd_unlearn_finetune_acc)))
                        print('SGD unlearn finetune acc std: ' + str(np.std(sgd_unlearn_finetune_acc)))
                        np.save('./result/SGD/'+str(self.args.dataset)+'/paint_unlearning_sigma/sgd_acc_finetune_b'+str(batch_size)+'_sigma'+str(sigma)+'_step'+str(sgd_step+1)+'.npy', sgd_unlearn_finetune_acc)
                    
        elif self.args.paint_utility_epsilon:
            epsilon_list = [0.1, 0.5, 1, 2, 5]
            batch_size_list = [32, 64, 128, 256, 0]
            num_remove_list = [1, 50, 100]
            create_nested_folder('./result/SGD/'+str(self.args.dataset)+'/paint_utility_epsilon/')
            for batch_size in batch_size_list:
                '''accuracy_scratch_D, mean_time, w_list = self.get_mean_performance(self.X_train, self.y_train, self.args.burn_in, self.args.sigma, 
                                                                              None, self.args.projection, batch_size, self.batch_idx, 
                                                                              len_list = 1, return_w = True, )
                np.save('./result/LMC/'+str(self.args.dataset)+'/paint_utility_epsilon/w_from_scratch_b'+str(batch_size)+'.npy', w_list)
                np.save('./result/LMC/'+str(self.args.dataset)+'/paint_utility_epsilon/acc_scratch_D_b'+str(batch_size)+'.npy', accuracy_scratch_D)'''
                # calculate K
                K_dict, _ = self.search_finetune_step(self.args.sigma, epsilon_list, batch_size_list)
                import pdb; pdb.set_trace()
                np.save('./result/LMC/'+str(self.args.dataset)+'/paint_utility_epsilon/K_list.npy', K_dict)
                for remove_idx, num_remove in enumerate(num_remove_list):
                    K_list = []
                    for epsilon in epsilon_list:
                        X_train_removed, y_train_removed = self.get_removed_data(num_remove_list[remove_idx])
                        accuracy_finetune, mean_time = self.get_mean_performance(X_train_removed, y_train_removed, K_dict[num_remove_list[remove_idx]][epsilon], self.args.sigma, w_list)
                        create_nested_folder('./result/LMC/'+str(self.args.dataset)+'/paint_utility_epsilon/'+str(num_remove)+'/')
                        np.save('./result/LMC/'+str(self.args.dataset)+'/paint_utility_epsilon/'+str(num_remove)+'/acc_finetune_epsilon'+str(epsilon)+'.npy', accuracy_finetune)
                        K_list.append(K_dict[num_remove_list[0]][epsilon])

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


    def epsilon_with_alpha_z(self, sigma, alpha, K, b, z):
        if b == 0:
            b = self.n
        c = 1-self.eta*self.m
        return alpha * z**2 / (2 * self.eta * sigma**2) * (c**2 - 1) / (1 - c**(-2 * K * self.n / b))
    
    def Z_B_sequential(self, step, b, previous_Zb):
        if b == 0:
            b = self.n
        if step == 1:
            return self.Z_B_loose(b)
        else:
            c = 1-self.eta*self.m
            return previous_Zb * c**(self.k_list[step-1] * self.n / b) + self.Z_B_loose(b)

    def epsilon_alpha_loose(self, sigma, alpha, K, b):
        if b == 0:
            b = self.n
        c = 1-self.eta*self.m
        return alpha * self.Z_B_loose(b)**2 / (2 * self.eta * sigma**2) * (c**2 - 1) / (1 - c**(-2 * K * self.n / b))
    
    def epsilon1_alpha_loose_nonconvergent(self, sigma, alpha, T, b, R):
        if b == 0:
            b = self.n
        c = 1-self.eta*self.m
        return alpha * (2*R)**2 / (2 * self.eta * sigma**2) * (c**2 - 1) / (1 - c**(-2 * T * self.n / b))
    
    def epsilon2_alpha_loose_nonconvergent(self, sigma, alpha, K, T, b, R):
        if b == 0:
            b = self.n
        c = 1-self.eta*self.m
        part1 = alpha * self.Z_B_loose(b)**2 / (2 * self.eta * sigma**2) * (c**2 - 1) / (1 - c**(-2 * K * self.n / b))
        part2 = 2*R*c**(2 * T * self.n / b)
        return part1 + part2
    
    def epsilon_alpha_loose_nonconvergent(self, sigma, alpha, q, K, T, b, R):
        p = 1/(1 - 1/q)
        if b == 128:
            if T > 30:
                T = 30
        elif b == self.n:
            T = 3000
        part1 = self.epsilon1_alpha_loose_nonconvergent(sigma, q*(alpha-1/p), T, b, R)
        part2 = self.epsilon2_alpha_loose_nonconvergent(sigma, p*alpha, K, T, b, R)
        return part1 + part2 * (alpha - 1/p) / (alpha - 1)
    
    def Z_B_loose(self, b):
        if b == 0:
            b = self.n
        c = 1-self.eta*self.m
        return 1/(1-c) * 2 * self.eta * self.M /b

    def Z_B(self,j,b):
        c = 1-self.eta*self.m
        return 1/(1-c) * c**(self.n/b-j-1) * 2 * self.eta * self.M /b
    
    def compute_k_loose(self, sigma, target_epsilon, b):
        k = 1
        epsilon = lambda alpha: (self.epsilon_alpha_loose(sigma, alpha,k, b)+ np.log(self.n)/(alpha-1))
        min_epsilon = minimize_scalar(epsilon, bounds=(1, 100000), method='bounded')
        while min_epsilon.fun > target_epsilon:
            k = k + 1
            epsilon = lambda alpha: (self.epsilon_alpha_loose(sigma, alpha,k, b)+ np.log(self.n)/(alpha-1))
            min_epsilon = minimize_scalar(epsilon, bounds=(1, 100000), method='bounded')
        
        #print(f'batch = {b}, epsilon={min_epsilon.fun}, alpha={min_epsilon.x}, loose K={k}')
        return k, min_epsilon.x

    def search_alpha(self, target_k, epsilon, batch_size, lower = 1e-15, upper = 10.0):
        if batch_size == 0:
            batch_size = self.n
        if self.compute_k_loose(lower, epsilon, batch_size)[0] < target_k or self.compute_k_loose(upper, epsilon, batch_size)[0] > target_k:
            print('not good upper lowers')
            return
        while upper - lower > 1e-8:
            mid = (lower + upper) / 2
            k, _ = self.compute_k_loose(mid, epsilon, batch_size)
            if k <= target_k:
                upper = mid
            else:
                lower = mid
        return upper
    
    def compute_k_loose_nonconvergent(self, sigma, target_epsilon, T, q, b, R):
        
        k = 1
        epsilon = lambda alpha: (self.epsilon_alpha_loose_nonconvergent(sigma, alpha, q, k, T, b, R)+ np.log(self.n)/(alpha-1))
        min_epsilon = minimize_scalar(epsilon, bounds=(1, 10000), method='bounded')
        while min_epsilon.fun > target_epsilon:
            k = k + 1
            epsilon = lambda alpha: (self.epsilon_alpha_loose_nonconvergent(sigma, alpha, q, k, T, b, R)+ np.log(self.n)/(alpha-1))
            min_epsilon = minimize_scalar(epsilon, bounds=(1, 10000), method='bounded')
        return k, min_epsilon.x
    
    def search_alpha_nonconvergent(self, target_k, epsilon, batch_size, T, R, q = 100, lower = 1e-15, upper = 10.0):
        if batch_size == 0:
            batch_size = self.n
        
        if self.compute_k_loose_nonconvergent(lower, epsilon, T, q, batch_size, R)[0] < target_k or self.compute_k_loose_nonconvergent(upper, epsilon, T, q, batch_size, R)[0] > target_k:
            print('not good upper lowers')
            return
        while upper - lower > 1e-8:
            mid = (lower + upper) / 2
            k, _ = self.compute_k_loose_nonconvergent(mid, epsilon, T, q, batch_size, R)
            if k <= target_k:
                upper = mid
            else:
                lower = mid
        return upper
    
    def search_finetune_step(self, sigma, epsilon_list, batch_list):
        K_dict = {}
        alpha_dict = {}
        for batch_size in batch_list:
            if batch_size == 0:
                batch_size = self.n
            K_list = {}
            alpha_list = {}
            for target_epsilon in epsilon_list:
                K, alpha = self.compute_k_loose(sigma, target_epsilon, batch_size)
                K_list[target_epsilon] = K
                alpha_list[target_epsilon] = alpha
            K_dict[batch_size] = K_list
            alpha_dict[batch_size] = alpha_list
        return K_dict, alpha_dict
    

    def get_mean_performance(self, X, y, step, sigma, w_list, projection, batch_size, batch_idx, len_list = 1, return_w = False, num_trial = 100):
        new_w_list = []
        trial_list = []
        time_list = []
        if w_list is None:
            for trial_idx in tqdm(range(num_trial)):
                w_init, time = self.run_stochastic_gradient_descent(None, X, y, step, sigma, len_list, 
                                                                    projection = projection, batch_size = batch_size, batch_idx = batch_idx[trial_idx])
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
                                                                   projection=projection, batch_size=batch_size, batch_idx = batch_idx[trial_idx])
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
    parser.add_argument('--lam', type=float, default=1e-6, help='L2 regularization')
    parser.add_argument('--num-removes', type=int, default=1000, help='number of data points to remove')
    parser.add_argument('--num-steps', type=int, default=10000, help='number of optimization steps')
    parser.add_argument('--train-mode', type=str, default='binary', help='train mode [ovr/binary]')
    parser.add_argument('--M', type = float, default = 1, help = 'set M-Lipschitz constant (norm of gradient)')
    parser.add_argument('--projection', type = float, default = 100.0, help = 'set the weight projection radius')
    parser.add_argument('--batch_size', type = int, default = 0, help = 'the batch size')

    parser.add_argument('--gpu', type = int, default = 6, help = 'gpu')
    parser.add_argument('--sigma', type = float, default = 0.03, help = 'the parameter sigma')
    parser.add_argument('--burn_in', type = int, default = 1000, help = 'burn in step number of LMC')

    parser.add_argument('--search_burnin', type = int, default = 0, help = 'whether grid search to paint for burn-in')
    parser.add_argument('--search_batch', type = int, default = 0, help = 'paint the batch size utility relation')
    parser.add_argument('--paint_utility_s', type = int, default = 0, help = 'paint the utility - s figure')
    parser.add_argument('--paint_utility_epsilon', type = int, default = 0, help = 'paint utility - epsilon figure')
    parser.add_argument('--paint_unlearning_sigma', type = int, default = 0, help = 'paint unlearning utility - sigma figure')
    parser.add_argument('--how_much_retrain', type = int, default = 0, help = 'supplementary for unlearning sigma')
    parser.add_argument('--compare_baseline', type = int, default = 0, help = 'compare with baseline')
    parser.add_argument('--compare_baseline_nonconvergent', type = int, default = 0, help = 'compare with the baselines with nonconvergent calculation')
    parser.add_argument('--sequential', type = int, default = 0, help = 'sequential unlearni')
    args = parser.parse_args()
    print(args)

    runner = Runner(args)
    runner.get_metadata()

    #import pdb; pdb.set_trace()

    runner.train()




if __name__ == '__main__':
    main()