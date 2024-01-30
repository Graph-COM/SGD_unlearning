# The script below are for SGD painting

# search burn in
#nohup python -u main_sgd.py --lam 1e-6 --dataset MNIST --search_burnin 1 --gpu 6 >./MNIST_LMC_search_burnin_lam1e6.log 2>&1 </dev/null &
#nohup python -u main_sgd.py --lam 1e-6 --dataset CIFAR10 --search_burnin 1 --gpu 7 >./CIFAR10_LMC_search_burnin_lam1e6.log 2>&1 </dev/null &
#nohup python -u main_sgd.py --lam 1e-7 --dataset MNIST_multiclass --search_burnin 1 --gpu 6 >./MNIST_multiclass_search_burnin_lam1e7.log 2>&1 </dev/null &

# search batch size and utility
nohup python -u main_sgd.py --lam 1e-6 --sigma 0.03 --dataset MNIST --projection 0 --search_batch 1 --gpu 0 >./MNIST_LMC_search_batch_lam1e6.log 2>&1 </dev/null &
nohup python -u main_sgd.py --lam 1e-6 --sigma 0.03 --dataset CIFAR10 --projection 0 --search_batch 1 --gpu 1 >./CIFAR10_LMC_search_batch_lam1e6.log 2>&1 </dev/null &

# compare with LMC and D2D baseline nonconvergent
#nohup python -u main_sgd.py --lam 1e-6 --dataset MNIST --projection 0 --compare_baseline_nonconvergent 1 --gpu 0 >./MNIST_SGD_compare_baseline_nonconvergent.log 2>&1 </dev/null &
#nohup python -u main_sgd.py --lam 1e-6 --dataset CIFAR10 --projection 0 --compare_baseline_nonconvergent 1 --gpu 2 >./CIFAR10_SGD_compare_baseline_nonconvergent.log 2>&1 </dev/null &

# compare with LMC and D2D baseline
#nohup python -u main_sgd.py --lam 1e-6 --dataset MNIST --projection 0 --compare_baseline 1 --gpu 4 >./MNIST_SGD_compare_baseline.log 2>&1 </dev/null &
#nohup python -u main_sgd.py --lam 1e-6 --dataset CIFAR10 --projection 0 --compare_baseline 1 --gpu 5 >./CIFAR10_SGD_compare_baseline.log 2>&1 </dev/null &

# compare sequential unlearning removal
#nohup python -u main_sgd.py --lam 1e-6 --dataset MNIST --projection 0 --sequential 1 --gpu 6 >./MNIST_SGD_sequential_32_64.log 2>&1 </dev/null &
#nohup python -u main_sgd.py --lam 1e-6 --dataset CIFAR10 --projection 0 --sequential 1 --gpu 7 >./CIFAR10_SGD_sequential_32_64.log 2>&1 </dev/null &

# paint unlearning unlearning - sigma figure
#nohup python -u main_sgd.py --lam 1e-6 --dataset MNIST --projection 0 --paint_unlearning_sigma 1 --gpu 0 >./MNIST_SGD_paint_unlearning_sigma_001.log 2>&1 </dev/null &
#nohup python -u main_sgd.py --lam 1e-6 --dataset CIFAR10 --projection 0 --paint_unlearning_sigma 1 --gpu 1 >./CIFAR10_SGD_paint_unlearning_sigma_001.log 2>&1 </dev/null &


# paint utility - epsilon figure
#nohup python -u main_sgd.py --lam 1e-6 --dataset MNIST --paint_utility_epsilon 1 --gpu 1 >./MNIST_SGD_paint_utility_epsilon.log 2>&1 </dev/null &
#nohup python -u main_sgd.py --lam 1e-6 --dataset CIFAR10 --paint_utility_epsilon 1 --gpu 6 >./CIFAR10_SGD_paint_utility_epsilon.log 2>&1 </dev/null &

# paint utility - s figure
#nohup python -u main_lmc.py --lam 1e-6 --sigma 0.03 --dataset MNIST --paint_utility_s 1 --gpu 1 >./MNIST_LMC_paint_utility_s.log 2>&1 </dev/null &
#nohup python -u main_lmc.py --lam 1e-6 --sigma 0.03 --dataset CIFAR10 --paint_utility_s 1 --gpu 6 >./CIFAR10_LMC_paint_utility_s.log 2>&1 </dev/null &





# how much retrain
#nohup python -u main_lmc.py --lam 1e-6 --dataset MNIST --how_much_retrain 1 --gpu 6 >./MNIST_how_much_retrain.log 2>&1 </dev/null &
#nohup python -u main_lmc.py --lam 1e-6 --dataset CIFAR10 --how_much_retrain 1 --gpu 7 >./CIFAR10_how_much_retrain.log 2>&1 </dev/null &


# calculate unlearning step between our bound and the baseline bound
#nohup python -u main_lmc.py --lam 1e-6 --dataset MNIST --compare_k 1 --gpu 2 >./MNIST_LMC_compare_k.log 2>&1 </dev/null &

# find the best batch size b per gradient for sgd
#nohup python -u main_lmc.py --lam 1e-6 --dataset MNIST --find_best_batch 1 --gpu 6 >./MNIST_LMC_find_best_batch.log 2>&1 </dev/null &

