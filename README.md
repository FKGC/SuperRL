# MRL-SNE
Source code for IJCAI2024 paper: Metric-Based Relational Learning with Selective Neighbor Entities for Few-Shot Knowledge Graph Completion

Few-shot Knowledge Graph (KG) completion is a focus of current research, where each task aims at querying unseen facts of a relation given few-shot reference entity pairs. However, existing works overlook two catergories of neighbor entities significant to few-shot KG completion. In this work, we propose a cascade neural network MRL-SNE, where we design neighbor entity encoders to identify these crucial neighbor entities. Evaluation in link prediction on two public datasets shows that our approach achieves new state-of-the-art results with different few-shot sizes.

# Requirements

```
python 3.6
Pytorch == 1.13.1
CUDA: 11.6
GPU: V100
```

# Datasets

We adopt Nell-One and Wiki-One datasets to evaluate our model, MRL-SNE.
The orginal datasets and pretrain embeddings are provided from [xiong's repo](https://github.com/xwhan/One-shot-Relational-Learning). 
For convenience, the datasets can be downloaded from [Nell data](https://sites.cs.ucsb.edu/~xwhan/datasets/nell.tar.gz)
and [Wiki data](https://sites.cs.ucsb.edu/~xwhan/datasets/wiki.tar.gz). 
The pre-trained embeddings can be downloaded from [Nell embeddings](https://drive.google.com/file/d/1XXvYpTSTyCnN-PBdUkWBXwXBI99Chbps/view?usp=sharing)
 and [Wiki embeddings](https://drive.google.com/file/d/1_3HBJde2KVMhBgJeGN1-wyvW88gRU1iL/view?usp=sharing).
Note that all these files were provided by xiong and we just select what we need here. 
All the dataset files and the pre-trained TransE embeddings should be put into the directory ./data/NELL and ./data/Wiki, respectively.

# How to run
For optimal performance, please train MRL-SNE as follows:

#### Nell-One

```
python main.py --fine_tune --num_layers 3 --lr 8e-5 --few 5 --early_stop_epoch 15 --prefix new_final_c1n1-3_5_NELL
```

#### Wiki-One

```
python main.py --fine_tune --eval_every 5000 --datapath "data/Wiki/" --num_layers 6 --lr 2e-4 --few 5 --early_stop_epoch 20 --prefix new_final_c1n1-6_5_Wiki
```

To test the trained models, please run as follows:

#### Nell-One

```
python main.py --test --num_layers 3 --lr 8e-5 --few 5 --prefix new_final_c1n1-3_5_NELL
```

#### Wiki-One

```
python main.py --test --datapath "data/Wiki/" --num_layers 6 --lr 2e-4 --prefix new_final_c1n1-6_5_Wiki
```

Here are explanations of some important args,

```bash
--data_path: "directory of dataset"
--few:       "the number of few in {few}-shot, as well as instance number in support set"
--num_layers:    "the number of enhancement layer"
--prefix:    "given name of current experiment"
--fine_tune  "whether to fine tune the pre_trained embeddings"
--device:    "the GPU number"
```

Normally, other args can be set to default values. See ``params.py`` for more details about argus if needed.

