# SGD_Unlearning

This is the official implementation of **Neurips 2024** paper 

[Certified Machine Unlearning via Noisy Stochastic Gradient Descent](https://arxiv.org/abs/2403.17105)

By Eli Chien, Haoyu Wang, Ziang Chen and Pan Li.

Please also check our **NeurIPS 2024 Spotlight** paper

[Langevin Unlearning: A New Perspective of Noisy Gradient Descent for Machine Unlearning](https://arxiv.org/abs/2401.10371)

## Environment requirements

The code is runnable under the following enveironment:

````
matplotlib                      3.7.2
notebook                        7.0.7
numpy                           1.24.4
pandas                          2.0.3
scikit-learn                    1.3.0
scipy                           1.10.1
seaborn                         0.13.0
torch                           2.0.0+cu117
torchvision                     0.15.1+cu117
tqdm                            4.65.0
````

## To implement and re-produce the result in Figure 3.a, run

````
python main_sgd.py --lam 1e-6 --dataset [MNIST/CIFAR10] --projection 0 --compare_baseline_nonconvergent 1
````

## To implement and re-produce the result in Figure 3.b, run

````
python main_sgd.py --lam 1e-6 --dataset [MNIST/CIFAR10] --projection 0 --sequential 1
````

## To implement and re-produce the result in Figure 3.c.d, run

````
python main_sgd.py --lam 1e-6 --dataset [MNIST/CIFAR10] --projection 0 --paint_unlearning_sigma 1
````

## Note

* use --gpu to allocate to a GPU device

* /result/SGD saves the results we run and report in our paper


## Citation

If you find our work useful, please cite us:
```
@article{chien2024stochastic,
  title={Stochastic Gradient Langevin Unlearning},
  author={Chien, Eli and Wang, Haoyu and Chen, Ziang and Li, Pan},
  journal={arXiv preprint arXiv:2403.17105},
  year={2024}
}
```
