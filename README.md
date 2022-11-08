### This repo presents some interesting examples of how to use pruning approach to accelerate CNN models.

### In addition, Knowledge distilation is used to recovery the prediction power of pruned model based on high accuracy pretrained teacher model


#### 1) The first example is based on [NNI framework](https://nni.readthedocs.io/en/stable/index.html) which presents a collection of automatic pruner tools, as well as tools for quantization (not considered here). 

`pip install nni`

- In the notebook `LotteryTicket_pruner.ipynb`, the Lottery Ticket approach based on paper [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/abs/1803.03635) was used to find the best pruned model for Resnet50 with dataset CIFAR10.

- With this example the one can play with different pruning policies into lottery ticket method, such as: [L1 or L2 norms](https://arxiv.org/abs/1608.08710) to rank filters, [FPGM](https://arxiv.org/abs/1811.00250) - Filter Pruning via Geometric Median, [Slim](https://arxiv.org/abs/1708.06519) - based on pruning scaling factors in BN layers, and [ADMM]() - Alternating Direction Method of Multipliers (solving 2 subproblems iteratively using gradient descent and Euclidean projection).


Examples of results:

| **Method** | **Acc_pruned** | **Acc_KD** |
| ------ | ------ | ------ |
| Baseline |  77.62% | 77.62% |
| LT - FPGM | 22.27% | 73.86% |
| LT - ADMM | 21.89% | 73.67% | 


Obs.: It is essential play/test others pruners tools from NNI. Furthermore, combining pruning + quantization remains open for investigations.


#### 2) The second tool tested here is the [torch_pruning](https://github.com/VainF/Torch-Pruning) providing a simple tool for getting the model dependence and then pruning filters for each layer.

`pip install torch-pruning`

or

`git clone https://github.com/VainF/Torch-Pruning.git`


_Next update will contain:_

- In the notebook `Iterative_pruner.ipynb`, a iterative pipeline is proposed to prune Resnet models and recovering accuracy by KD.

- Comparison of different policies to choose filters to prune.  

- A backward pruning pipeline gives better compression ratio without drastic accuracy drop. 

Examples of results:

| **Method** | **Acc_pruned** | **Acc_KD** |
| ------ | ------ | ------ |
| Baseline |  |  |
| IteB - L1 |  |  |
| IteB - Rand |  |  |
| IteB - Clustering |  |  | 



**Credits:**

`LotteryTicket_pruner.ipynb`: contains myself code + essential codes from NNI examples files.

`Iterative_pruner.ipynb`: own reverse pipeline to prune Resnet models.
