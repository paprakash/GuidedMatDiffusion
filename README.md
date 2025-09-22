# For Pre-training of foundation model, follow the original DiffCSP method to do training on you data

Implementation codes for Crystal Structure Prediction by Joint Equivariant Diffusion (DiffCSP) for pre-training a foundation model.

[**[DiffCSP Paper]**](https://arxiv.org/abs/2309.04475)


### Setup

Clone this repo, cd into its directory, make sure you are on pre_training branch, and run
```
pip install -e .
```
or 
```
pip install git+https://github.com/paprakash/GuidedMatDiffusion.git
```
The former is the preferred way, as one will still need scripts and configuration files which are present in this repo if installing directly.

torch-scatter and torch-sparse should also be installed. Their installation will depend on the version of PyTorch which is installed. 
For example, to install the binaries for PyTorch 2.3.0, simply run

```
pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.3.0+${CUDA}.html
```

where `${CUDA}` should be replaced by either `cpu`, `cu118`, or `cu121` depending on your PyTorch installation.

|             | `cpu` | `cu118` | `cu121` |
| ----------- | ----- | ------- | ------- |
| **Linux**   | ✅     | ✅       | ✅       |
| **Windows** | ✅     | ✅       | ✅       |
| **macOS**   | ✅     |         |         |

Rename the `.env.template` file into `.env` , specify the below variables and source it.

```
PROJECT_ROOT: the absolute path of this repo
HYDRA_JOBS: the absolute path to save hydra outputs
WABDB_DIR: the absolute path to save wabdb outputs
```

### Training

For the Ab Initio Generation task

```
python diffcsp/run.py data=alex_20 model=diffusion_w_type expname=<expname>
```

The ``<expname>`` tag can be an arbitrary name to identify each experiment. The dataset 'alex_20' can be found on [huggingface_alex](https://huggingface.co/datasets/paprakash/GuidedMatDiffusion_data/tree/main/alex_20). The pre-trained checkpoint is also provided on [huggingface_ckpt](https://huggingface.co/paprakash/GuidedMatDiffusion_model/tree/main/foundation_model).

If one does not want to use WandB during training, comment out the "wandb" section in conf/logging/default.yaml. 



### Acknowledgments

The main framework of this codebase is build upon [DiffCSP](https://github.com/jiaor17/DiffCSP.git). For pre-training alexandria dataset is from [Alexandria](https://alexandria.icams.rub.de).

### Original Citation

```
@article{
  
}
```

