from hmac import new
import time
import numpy as np
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
from scipy.optimize import minimize_scalar

import torch
import torch.nn.functional as F

from utils import load_features, create_nested_folder, transform_array
from sgd import stochastic_gradient_descent_algorithm_multiclass


class Runner():
    def __init__(self, args):
        self.device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')
        self.args = args
        if args.dataset == 'CIFAR10_MULTI':
            self.X_train, self.X_test, self.y_train, self.y_train_onehot, self.y_test = load_features(args)
            self.X_train = self.X_train[:48640]
            self.y_train = self.y_train[:48640]
            self.dim_w = 512
            self.num_class = 10
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
        self.L = 1 + self.args.lam * self.n
        self.L = self.L
        print('L smooth constant'+str(self.L))
        # m-strongly convex constant
        self.m = self.args.lam * self.n
        self.m = self.m
        print('m strongly convex:'+str(self.m))
        # M-Lipschitz constant
        self.M = self.args.M
        print('M lipschitz constant:'+str(self.M))
        # calculate step size
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
        if self.args.compare_baseline_nonconvergent:
            # compare with the baseline nonconvergent (remove 1 data see sigma and utility)
            epsilon_list = [0.05, 0.1, 0.5, 1, 2, 5]
            batch_list = [512, 0]
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
                        sigma_list.append(self.search_alpha_nonconvergent(target_k, target_epsilon, batch_size, burn_in, self.projection, 2))
                    print('batch: '+str(batch_size)+'target k:'+str(target_k) + ' sigma: '+str(sigma_list))
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
        elif self.args.retrain_noiseless == 1:
            num_remove_list = [1, 10, 50, 100, 500, 1000] # the number of data to remove
            for num_remove in num_remove_list:
                create_nested_folder('./result/SGD/'+str(self.args.dataset)+'/retrain_noiseless/')
                X_train_removed, y_train_removed = self.get_removed_data(num_remove)
                accuracy_scratch_Dnew, mean_time = self.get_mean_performance(X_train_removed, y_train_removed, self.args.burn_in, 0, None)
                np.save('./result/SGD/'+str(self.args.dataset)+'/retrain_noiseless/retrain_noiseless'+str(num_remove)+'.npy', accuracy_scratch_Dnew)
        else:
            print('check!')

    def get_removed_data(self, num_remove):
        X_train_removed = self.X_train[:-num_remove,:]
        y_train_removed = self.y_train[:-num_remove]
        new_X_train = torch.randn(num_remove, self.dim_w)
        norms = new_X_train.norm(dim=1, keepdim=True)
        new_X_train = new_X_train / norms
        new_X_train = new_X_train.to(self.device)
        new_y_train = torch.randint(0, 10, (1, num_remove)).reshape(-1)
        new_y_train = transform_array(new_y_train)
        new_y_train = new_y_train.to(self.device)
        X_train_removed = torch.cat((X_train_removed, new_X_train), 0)
        y_train_removed = torch.cat((y_train_removed, new_y_train), 0)
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
        return alpha * self.Z_B_loose(b)**2 / (2 * self.eta * sigma**2) * c**(2 * K * self.n / b)
    
    def epsilon1_alpha_loose_nonconvergent(self, sigma, alpha, T, b, R):
        if b == 0:
            b = self.n
        c = 1-self.eta*self.m
        return alpha * (2*R)**2 / (2 * self.eta * sigma**2) * c**(2 * T * self.n / b)
    
    def epsilon2_alpha_loose_nonconvergent(self, sigma, alpha, K, T, b, R):
        if b == 0:
            b = self.n
        c = 1-self.eta*self.m
        part1 = self.Z_B_loose(b)
        part2 = 2*R*c**(T * self.n / b)
        ans = alpha * (part1+part2)**2 / (2 * self.eta * sigma**2) * c**(2 * K * self.n / b)
        return ans
    
    def epsilon_alpha_loose_nonconvergent(self, sigma, alpha, q, K, T, b, R):
        p = 1/(1 - 1/q)
        if b == 128:
            if T > 30:
                T = 30
        elif b == self.n:
            T = 3000
        part1 = self.epsilon1_alpha_loose_nonconvergent(sigma, q*(alpha), T, b, R)
        part2 = self.epsilon2_alpha_loose_nonconvergent(sigma, p*alpha, K, T, b, R)
        return (part1 + part2) * (alpha - 1/p) / (alpha - 1)
    
    def Z_B_loose(self, b):
        if b == 0:
            b = self.n
        c = 1-self.eta*self.m
        return 1/(1-c**(self.n/b)) * 2 * self.eta * self.M /b

    def Z_B(self,j,b):
        c = 1-self.eta*self.m
        return 1/(1-c**(self.n/b)) * c**(self.n/b-j-1) * 2 * self.eta * self.M /b
    
    def compute_k_loose(self, sigma, target_epsilon, b):
        k = 1
        epsilon = lambda alpha: (self.epsilon_alpha_loose(sigma, alpha,k, b)+ np.log(self.n)/(alpha-1))
        min_epsilon = minimize_scalar(epsilon, bounds=(2, 100000), method='bounded')
        while min_epsilon.fun > target_epsilon:
            k = k + 1
            epsilon = lambda alpha: (self.epsilon_alpha_loose(sigma, alpha,k, b)+ np.log(self.n)/(alpha-1))
            min_epsilon = minimize_scalar(epsilon, bounds=(2, 100000), method='bounded')
        
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
        min_epsilon = minimize_scalar(epsilon, bounds=(2, 10000), method='bounded')
        while min_epsilon.fun > target_epsilon:
            k = k + 1
            epsilon = lambda alpha: (self.epsilon_alpha_loose_nonconvergent(sigma, alpha, q, k, T, b, R)+ np.log(self.n)/(alpha-1))
            min_epsilon = minimize_scalar(epsilon, bounds=(2, 10000), method='bounded')
        return k, min_epsilon.x
    
    def search_alpha_nonconvergent(self, target_k, epsilon, batch_size, T, R, q = 2, lower = 1e-15, upper = 10.0):
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
    

    def get_mean_performance(self, X, y, step, sigma, w_list, projection, batch_size, batch_idx, len_list = 1, return_w = False, num_trial = 100):
        new_w_list = []
        trial_list = []
        time_list = []
        if w_list is None:
            for trial_idx in tqdm(range(num_trial)):
                w_init, time = self.run_stochastic_gradient_descent(None, X, y, step, sigma, len_list, 
                                                                    projection = projection, batch_size = batch_size, batch_idx = batch_idx[trial_idx])
                time_list.append(time)
                w_init = np.vstack(w_init)
                new_w_list.append(w_init)
                accuracy = self.test_accuracy(w_init)
                trial_list.append(accuracy)
        else:
            for trial_idx in tqdm(range(num_trial)):
                w = w_list[trial_idx]
                w = torch.tensor(w)
                new_w, time = self.run_stochastic_gradient_descent(w, X, y, step, sigma, len_list = 1,
                                                                   projection=projection, batch_size=batch_size, batch_idx = batch_idx[trial_idx])
                time_list.append(time)
                new_w = np.vstack(new_w)
                new_w_list.append(new_w)
                accuracy = self.test_accuracy(new_w)
                trial_list.append(accuracy)
        mean_time = np.mean(time_list)

        if return_w:
            new_w_list = np.stack(new_w_list, axis=0)
            return trial_list, mean_time, new_w_list
        else:
            return trial_list, mean_time
                
    def test_accuracy(self, w_list):
        w = torch.tensor(w_list)
        # test accuracy (before removal)
        pred = self.X_test.matmul(w)
        score = F.softmax(pred, dim = -1)
        _, pred_class = score.max(dim = 1)
        _, label = self.y_test.max(dim = 1)
        correct_predictions = (pred_class == label).sum().item()
        accuracy = correct_predictions / score.shape[0]
        return accuracy
    def run_stochastic_gradient_descent(self, init_point, X, y, burn_in, sigma, len_list, projection, batch_size, batch_idx):
        start_time = time.time()
        w_list = stochastic_gradient_descent_algorithm_multiclass(init_point, self.dim_w, X, y, self.args.lam*self.n, sigma = sigma, 
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
    parser.add_argument('--train-mode', type=str, default='multi', help='train mode [ovr/binary]')
    parser.add_argument('--M', type = float, default = 1, help = 'set M-Lipschitz constant (norm of gradient)')
    parser.add_argument('--projection', type = float, default = 100.0, help = 'set the weight projection radius')
    parser.add_argument('--batch_size', type = int, default = 0, help = 'the batch size')

    parser.add_argument('--gpu', type = int, default = 6, help = 'gpu')
    parser.add_argument('--sigma', type = float, default = 0.03, help = 'the parameter sigma')
    parser.add_argument('--burn_in', type = int, default = 1000, help = 'burn in step number of LMC')

    parser.add_argument('--paint_unlearning_sigma', type = int, default = 0, help = 'paint unlearning utility - sigma figure')
    parser.add_argument('--compare_baseline_nonconvergent', type = int, default = 0, help = 'compare with the baselines with nonconvergent calculation')
    parser.add_argument('--sequential', type = int, default = 0, help = 'sequential unlearni')
    parser.add_argument('--retrain_noiseless', type = int, default = 0, help = 'retrain_noiseless')
    args = parser.parse_args()
    print(args)
    runner = Runner(args)
    runner.get_metadata()
    runner.train()

if __name__ == '__main__':
    main()