
<p align="center">
  <h1 align="center">Multi-HMR: Multi-Person Whole-Body Human Mesh Recovery in a Single Shot</h1>

  <p align="center">
    <a href="https://fabienbaradel.github.io/">Fabien Baradel*</a>, 
    <a href="https://europe.naverlabs.com/people_user_naverlabs/matthieu-armando/">Matthieu Armando</a>,  
    <a href="https://europe.naverlabs.com/people_user_naverlabs/Salma-Galaaoui/?asp_highlight=Salma+Galaaoui&p_asid=9">Salma Galaaoui</a>,  
    <a href="https://europe.naverlabs.com/people_user_naverlabs/Romain-Br%C3%A9gier/">Romain Brégier</a>,  <br>
    <a href="[./](https://europe.naverlabs.com/people_user_naverlabs/Philippe-Weinzaepfel/?asp_highlight=Philippe+Weinzaepfel&p_asid=9)">Philippe Weinzaepfel</a>, 
    <a href="https://europe.naverlabs.com/people_user_naverlabs/Gregory-Rogez/">Grégory Rogez</a>, 
    <a href="https://europe.naverlabs.com/people_user_naverlabs/Thomas-Lucas/">Thomas Lucas*</a> 
  </p>

  <p align="center">
    <sup>*</sup> equal contribution
  </p>

  <p align="center">
  <a href="https://arxiv.org/abs/2402.14654"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2402.14654-00ff00.svg"></a>
  <a href="https://europe.naverlabs.com/?p=9361171&preview=true"><img alt="Blogpost" src="https://img.shields.io/badge/Blogpost-up-yellow"></a>
  <a href="https://multihmr-demo.europe.naverlabs.com/"><img alt="Demo" src="https://img.shields.io/badge/Demo-up-blue"></a>
  <!-- <a href="./"><img alt="Hugging Face Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue"></a> -->
  </p>

  <p align="left">
  <a href="https://paperswithcode.com/sota/3d-human-reconstruction-on-ehf?p=multi-hmr-multi-person-whole-body-human-mesh"><img src="https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-hmr-multi-person-whole-body-human-mesh/3d-human-reconstruction-on-ehf"></a><br>
  <a href="https://paperswithcode.com/sota/3d-human-pose-estimation-on-ubody?p=multi-hmr-multi-person-whole-body-human-mesh"><img src="https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-hmr-multi-person-whole-body-human-mesh/3d-human-pose-estimation-on-ubody"></a><br>
  <a href="https://paperswithcode.com/sota/3d-multi-person-human-pose-estimation-on?p=multi-hmr-multi-person-whole-body-human-mesh"><img src="https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-hmr-multi-person-whole-body-human-mesh/3d-multi-person-human-pose-estimation-on"></a><br>
  <a href="https://paperswithcode.com/sota/3d-human-pose-estimation-on-3dpw?p=multi-hmr-multi-person-whole-body-human-mesh"><img src="https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-hmr-multi-person-whole-body-human-mesh/3d-human-pose-estimation-on-3dpw"></a><br>
  </p>

  <div align="center">
  <img width="49%" alt="Multi-HMR illustration 1" src="assets/visu1.gif">
  <img width="49%" alt="Multi-HMR illustration 2" src="assets/visu2.gif">

  <br>
  Multi-HMR is a simple yet effective single-shot model for multi-person and expressive human mesh recovery.
  It takes as input a single RGB image and efficiently performs 3D reconstruction of multiple humans in camera space.
  <br>
</div>
</p>

## Installation
First, you need to clone the repo.

We recommand to use virtual enviroment for running MultiHMR.
Please run the following lines for creating the environment with ```venv```:
```bash
python3.9 -m venv .multihmr
source .multihmr/bin/activate
pip install -r requirements.txt
```

Otherwise you can also create a conda environment.
```bash
conda env create -f conda.yaml
conda activate multihmr
```

The installation has been tested with python3.9 and CUDA 11.7.

Checkpoints will automatically be downloaded to `$HOME/models/multiHMR` the first time you run the demo code.

Besides these files, you also need to download the *SMPLX* model.
You will need the [neutral model](http://smplify.is.tue.mpg.de) for running the demo code.
Please go to the corresponding website and register to get access to the downloads section.
Download the model and place `SMPLX_NEUTRAL.npz` in `./models/smplx/`.

## Run Multi-HMR on images
The following command will run Multi-HMR on all images in the specified `--img_folder`, and save renderings of the reconstructions in `--out_folder`.
The `--model_name` flag specifies the model to use.
The `--extra_views` flags additionally renders the side and bev view of the reconstructed scene, `--save_mesh` saves meshes as in a '.npy' file.
```bash
python3.9 demo.py \
    --img_folder example_data \
    --out_folder demo_out \
    --extra_views 1 \
    --model_name multiHMR_896_L
```

## Pre-trained models
We provide multiple pre-trained checkpoints.
Here is a list of their associated features.
Once downloaded you need to place them into `$HOME/models/multiHMR`.

| modelname                     | training data                     | backbone | resolution | runtime (ms) |
|-------------------------------|-----------------------------------|----------|------------|--------------|
| [multiHMR_896_L](https://download.europe.naverlabs.com/ComputerVision/MultiHMR/multiHMR_896_L.pt)    | BEDLAM+AGORA+CUFFS+UBody                      | ViT-L    | 896x896    |    126       |

We compute the runtime on GPU V100-32GB.

## License
The code is distributed under the CC BY-NC-SA 4.0 License.\
See [Multi-HMR LICENSE](Multi-HMR_License.txt), [Checkpoint LICENSE](Checkpoint_License.txt) and [Example Data LICENSE](Example_Data_License.txt) for more information.

## Citing
If you find this code useful for your research, please consider citing the following paper:
```bibtex
@inproceedings{multi-hmr2024,
    title={Multi-HMR: Multi-Person Whole-Body Human Mesh Recovery in a Single Shot},
    author={Baradel*, Fabien and 
            Armando, Matthieu and 
            Galaaoui, Salma and 
            Br{\'e}gier, Romain and 
            Weinzaepfel, Philippe and 
            Rogez, Gr{\'e}gory and
            Lucas*, Thomas
            },
    booktitle={arXiv},
    year={2024}
}
```

