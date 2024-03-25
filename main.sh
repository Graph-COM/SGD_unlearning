# The script below are for SGD painting

# compare with LMC and D2D baseline nonconvergent
#nohup python -u main_sgd.py --lam 1e-6 --dataset MNIST --projection 0 --compare_baseline_nonconvergent 1 --gpu 6 >./MNIST_SGD_compare_baseline_nonconvergent.log 2>&1 </dev/null &
#nohup python -u main_sgd.py --lam 1e-6 --dataset CIFAR10 --projection 0 --compare_baseline_nonconvergent 1 --gpu 7 >./CIFAR10_SGD_compare_baseline_nonconvergent.log 2>&1 </dev/null &
#nohup python -u main_sgd_multiclass.py --lam 1e-6 --dataset CIFAR10_MULTI --projection 0 --compare_baseline_nonconvergent 1 --gpu 2 >./CIFAR10_MULTI_SGD_compare_baseline_nonconvergent.log 2>&1 </dev/null &

# compare sequential unlearning removal
#nohup python -u main_sgd.py --lam 1e-6 --dataset MNIST --projection 0 --sequential 1 --gpu 6 >./MNIST_SGD_sequential.log 2>&1 </dev/null &
#nohup python -u main_sgd.py --lam 1e-6 --dataset CIFAR10 --projection 0 --sequential 1 --gpu 7 >./CIFAR10_SGD_sequential.log 2>&1 </dev/null &

# paint unlearning unlearning - sigma figure
#nohup python -u main_sgd.py --lam 1e-6 --dataset MNIST --projection 0 --paint_unlearning_sigma 1 --gpu 0 >./MNIST_SGD_paint_unlearning_sigma_001.log 2>&1 </dev/null &
#nohup python -u main_sgd.py --lam 1e-6 --dataset CIFAR10 --projection 0 --paint_unlearning_sigma 1 --gpu 1 >./CIFAR10_SGD_paint_unlearning_sigma_001.log 2>&1 </dev/null &
