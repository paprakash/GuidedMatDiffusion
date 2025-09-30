# Guided Diffusion for the Discovery of New Superconductors 

Implemetation guided diffusion using property embedding and classifier free guidance

[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue)](https://huggingface.co/paprakash/GuidedMatDiffusion_model)
[![Hugging Face Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-blue)](https://huggingface.co/datasets/paprakash/GuidedMatDiffusion_data)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.3.0](https://img.shields.io/badge/PyTorch-2.3.0-EE4C2C.svg)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-PyTorch-792EE5.svg)](https://lightning.ai/docs/pytorch/stable/)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/jiaor17/DiffCSP/blob/main/LICENSE) 

 [**[Original Paper]**](https://www.arxiv.org/abs/2509.25186)

### Workflow Pipeline

<img src="fig/workflow.png" alt="workflow_pipeline" width="100%">

#### *De novo* generated new structural prototypes

<img src="fig/prototypes.png" alt="prototypes" width="80%">


### Setup

Clone this repo, cd into its directory, and run
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


Fine-tuning a pre-trained foundation model for the Ab Initio Generation task

```
python diffcsp/run.py data=supccomb_12 model=diffusion_w_type expname=<expname> train.ckpt_path=<path_to_pretrained_checkpoint>
```

The ``<dataset>`` tag for fine_tuning for superconductivity is supcomb_12 but other dataset can be selected from perov_5, mp_20, mpts_52 and carbon_24, and the ``<expname>`` tag can be an arbitrary name to identify each experiment. Trained checkpoint to generates superconductors as well as a pre-trained foundation model used for fine-tunning is provided at [hugging_face](https://huggingface.co/paprakash/GuidedMatDiffusion_model/tree/main).

If one does not want to use WandB during training, comment out the "wandb" section in conf/logging/<your_config>.yaml. 


### Ab initio generation

```
python scripts/generation.py --model_path <model_path> --dataset supccomb_12 --save_path <path_to_save_gen_structures> --band_gap <scaled/normalized_Tc> --guide_w <Guidace_weight> --batch_size <batch_size> --num_batch_to_sample <samples_in_each_batch>
```
To-do: changing the band_gap tag to property

Note: To scale the property, here T_c value (Tc_actual) you want to give the model use - 
```
scaler = torch.load(Path(scaler_path) / 'prop_scaler.pt')
Tc = scaler.transform(Tc_actual)
```

#### Pre-Training

Details of pre-training a foundation model, with alexandria dataset can be found in the pre_training branch.


### Acknowledgments

The main framework of this codebase is build upon [DiffCSP](https://github.com/jiaor17/DiffCSP.git).Fine-tuning superconductivity dataset is from combining [Cerqueira et al](https://archive.materialscloud.org/records/3kbt5-r3n56) and [Gibson et al](https://arxiv.org/pdf/2503.20005). For pre-training alexandria dataset is from [Alexandria](https://alexandria.icams.rub.de).


### Original Citation

```
@misc{https://doi.org/10.48550/arxiv.2509.25186,
  doi = {10.48550/ARXIV.2509.25186},
  url = {https://arxiv.org/abs/2509.25186},
  author = {Prakash,  Pawan and Gibson,  Jason B. and Li,  Zhongwei and Di Gianluca,  Gabriele and Esquivel,  Juan and Fuemmeler,  Eric and Geisler,  Benjamin and Kim,  Jung Soo and Roitberg,  Adrian and Tadmor,  Ellad B. and Liu,  Mingjie and Martiniani,  Stefano and Stewart,  Gregory R. and Hamlin,  James J. and Hirschfeld,  Peter J. and Hennig,  Richard G.},
  keywords = {Superconductivity (cond-mat.supr-con),  Materials Science (cond-mat.mtrl-sci),  Artificial Intelligence (cs.AI),  FOS: Physical sciences,  FOS: Physical sciences,  FOS: Computer and information sciences,  FOS: Computer and information sciences},
  title = {Guided Diffusion for the Discovery of New Superconductors},
  publisher = {arXiv},
  year = {2025},
  copyright = {Creative Commons Attribution 4.0 International}
}
```

