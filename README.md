# AS-GCN in Tensorflow

**Note:** This is a fork of the original [AS-GCN repository](https://github.com/huangwb/AS-GCN), adapted to a more recent version of Tensorflow and incorporating an ablation where attention is not used in the GCN classifier.

We provide Tensorflow implementations for the paper "Adaptive Sampling Towards Fast Graph Representation Learning". You need to install Tensorflow and other related python packages, e.g. scypy, networkx, etc. Our code is based on the orginal GCN framework, and takes inspirations from GraphSAGE and FastGCN. The core of this code is that we separate the sampling (i.e. sampler) and propagation (i.e. propagator) processes, both of which are implemented by tensorflow. 

Please note that it is possible that the results by this code would be slightly different to those reported in the paper due to the random noise and some post-modifications of the code.

## Installation

We provide an environment file for conda. You can create a conda environment by running
```sh
conda env create -f environment.yml
```

This will create an environment named `asgcn`. Additionally, the paths to the 
CUDA-related libraries need to be updated. To do this when activating the
environment, run

```sh
conda activate asgcn
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```

The above change requires deactivating and reactivating the environment for
the changes to take effect.

## Data Preparation
We provide the datasets including cora, citeseer, pubmed. For reddit, you need to download it from http://snap.stanford.edu/graphsage/ and then transfer it to the correct format using transfer_Graph.py.


## Run the Code

For cora, citeseer and pubmed, try
```
python run_pubmed.py --dataset dataset_name
```


## Citation
If you find this code useful for your research, please cite the paper:


```
@inproceedings{huang2018adapt,
  title={Adaptive Sampling Towards Fast Graph Representation Learning},
  author={Huang, Wenbing and Zhang, Tong and Rong, Yu and Huang, Junzhou},
  booktitle={Advances in Neural Information Processing Systems (NIPS)},
  pages={},
  year={2018}
}
```





