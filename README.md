# MRL-SNE
Source code for IJCAI2024 paper: Metric-Based Relational Learning with Selective Neighbor Entities for Few-Shot Knowledge Graph Completion

Few-shot Knowledge Graph (KG) completion is a focus of current research, where each task aims at querying unseen facts of a relation given few-shot reference entity pairs. However, existing works overlook two catergories of neighbor entities significant to few-shot KG completion. In this work, we propose a cascade neural network MRL-SNE, where we design neighbor entity encoders to identify these crucial neighbor entities. Evaluation in link prediction on two public datasets shows that our approach achieves new state-of-the-art results with different few-shot sizes.

# Requirements

```
python 3.6
Pytorch == 1.13.1
CUDA: 11.6
GPU: NVIDIA GeForce V100
```

# Datasets

We adopt Nell-One and Wiki-One datasets to evaluate our model, MRL-SNE.
The orginal datasets and pretrain embeddings are provided from [xiong's repo](https://github.com/xwhan/One-shot-Relational-Learning). 
For convenience, the datasets can be downloaded from [Nell data](https://sites.cs.ucsb.edu/~xwhan/datasets/nell.tar.gz)
and [Wiki data](https://sites.cs.ucsb.edu/~xwhan/datasets/wiki.tar.gz). 
The pre-trained embeddings can be downloaded from [Nell embeddings](https://drive.google.com/file/d/1XXvYpTSTyCnN-PBdUkWBXwXBI99Chbps/view?usp=sharing)
 and [Wiki embeddings](https://drive.google.com/file/d/1_3HBJde2KVMhBgJeGN1-wyvW88gRU1iL/view?usp=sharing).
Note that all these files were provided by xiong and we just select what we need here. 
All the dataset files and the pre-trained TransE embeddings should be put into the directory ./NELL and ./Wiki, respectively.

# How to run
We exclusively present comparisons with other baseline statistics in the 'Pre-train' setting, aligning with state-of-the-art benchmarks in the paper. Moreover, our code introduces an alternative setting termed 'In-train,' proposed by MetaR, in which our model surpasses established baseline methods as well. For optimal performance, please run as follows:

## Pre-train Setting

#### Nell-One

```
python main.py --fine_tune --form 'Pre-Train' --num_layers 3 --lr 8e-5 --few 5 --early_stop_epoch 15 --prefix new_final_c1n1-3_5_NELL_Pre_Train
```

#### Wiki

```
python main.py --fine_tune --form 'Pre-Train' --eval_every 5000 --datapath "data/Wiki/" --num_layers 6 --lr 2e-4 --few 5 --early_stop_epoch 20 --prefix new_final_c1n1-6_5_Wiki_Pre_Train
```

## In-train Setting

#### Nell-One

```
python main.py --fine_tune --form 'In-Train' --num_layers 3 --lr 8e-5 --few 5 --early_stop_epoch 15 --prefix new_final_c1n1-3_5_NELL_Pre_Train
```

#### Wiki

```
python main.py --fine_tune --form 'In-Train' --eval_every 5000 --datapath "data/Wiki/" --num_layers 6 --lr 2e-4 --few 5 --early_stop_epoch 20 --prefix new_final_c1n1-6_5_Wiki_Pre_Train
```

To test the trained models, please run as follows:

#### Nell

```
python main.py --test --num_layers 3 --lr 8e-5 --few 5 --prefix new_final_c1n1-3_5_NELL_Pre_Train
```

#### Wiki

```
python main.py --test --datapath "data/Wiki/" --num_layers 6 --lr 2e-4 --prefix new_final_c1n1-6_5_Wiki_Pre_Train
```

