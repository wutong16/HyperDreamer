
# De-rendering 3D Objects in the Wild
[**Paper**](https://arxiv.org/abs/2201.02279) |  [**Video**](https://youtu.be/IV5orKpwh80) | [**Project Page**](https://www.robots.ox.ac.uk/~vgg/research/derender3d/)

This is the official implementation for the CVPR 2022 paper:

> **De-rendering 3D objects in the Wild**
>
> [Felix Wimbauer](https://www.vision.in.tum.de/members/wimbauer)<sup>1,2</sup>, [Shangzhe Wu](https://elliottwu.com/)<sup>1</sup> and [Christian Rupprecht](https://chrirupp.github.io/)<sup>1</sup>  
> <sup>1</sup>Visual Geometry Group, University of Oxford, <sup>2</sup>Technical University of Munich
> 
> [**CVPR 2022** (arXiv)](https://arxiv.org/abs/2201.02279)

https://user-images.githubusercontent.com/16351108/167137537-c3ae52ae-aaac-41af-afcb-5aa282b415cd.mp4


A method for de-rendering a 3D object from a single image into shape, material, and lighting, that is trained in a weakly-supervised fashion relying only on rough shape estimates. 

## 📋 Abstract

With increasing focus on augmented and virtual reality applications (XR) comes the demand for algorithms that can lift 
objects from images and videos into representations that are suitable for a wide variety of related 3D tasks. 
Large-scale deployment of XR devices and applications means that we cannot solely rely on supervised learning, as 
collecting and annotating data for the unlimited variety of objects in the real world is infeasible. We present a 
weakly supervised method that is able to decompose a single image of an object into shape (depth and normals), material 
(albedo, reflectivity and shininess) and global lighting parameters. For training, the method only relies on a rough 
initial shape estimate of the training objects to bootstrap the learning process. This shape supervision can come for 
example from a pretrained depth network or - more generically - from a traditional structure-from-motion pipeline. In 
our experiments, we show that the method can successfully de-render 2D images into a decomposed 3D representation and 
generalizes to unseen object categories. Since in-the-wild evaluation is difficult due to the lack of ground truth data, 
we also introduce a photo-realistic synthetic test set that allows for quantitative evaluation.
```
@InProceedings{wimbauer2022rendering,
  title={De-rendering 3D Objects in the Wild},
  author={Wimbauer, Felix and Wu, Shangzhe and Rupprecht, Christian},
  booktitle={CVPR},
  year={2022}
}
```

## 🏗️️ Setup

### 🐍 Python Environment

We use Conda to manage our Python environment:
```shell
conda env create -f environment.yml
```
Then, activate the conda environment :
```shell
conda activate derender3d
```

### 📸 Checkpoints

We provide download links for pretrained models for **CelebA-HQ** and **Co3D**.
Models will be stored under `results/models` at the same location the training checkpoints will be stored.

```shell
setup/download_model.sh {celebahq|co3d}
```

### 💾 Processed Datasets

We provide download links for the processed **Co3D** dataset.
For **CelebA-HQ**, the licensing is unclear, which is why we can only provide intructions to reproduce the dataset.
Datasets will be stored under `datasets`.
If you should prefer another storage location, you can create soft-links to the respective locations in the `datasets` folder.

  
```shell
setup/download_processed_co3d.sh
```

## 🎤 Demo

**Coming Soon**

For now, please have a look at the `scripts` directory, which provides many useful code snippets for data inspection, 
image generation, relighting videos, and consistency videos.

## 🏋️ Training

We provide experiment configurations under `experiments/release` to reproduce the results we reported in the paper.
To perform training, run the following commands:

**CelebA-HQ**
```shell
python run.py --config experiments/release/celebahq.yml --num_workers 8 --gpu 0
python run.py --config experiments/release/celebahq_nr.yml --num_workers 8 --gpu 0
```

**Co3D**
```shell
python run.py --config experiments/release/co3d.yml --num_workers 8 --gpu 0
```

## 📊 Evaluation

To recalculate the numbers we report in the paper, please run the `scripts/eval_cosy.py` script.
This requires you to setup the **Co3D** checkpoint and **COSy** dataset, as explained before.
```shell
python scripts/eval_cosy.py
```

## Manual Dataset Creation

**Coming Soon**

## TODO

- [x] Check reproducibility
- [x] Refactor and clean up code
- [x] Create download scripts for data and trained models
- [x] Check conda environment
- [x] Write detailed ReadMe
- [ ] Create demo
- [ ] Create fork for **Unsup3D** with data setup scripts (for CelebA-HQ)
- [ ] Create fork for **Co3D** with data setup scripts

## Acknowledgements

This repository is largely based on the [Unsup3D repository](https://github.com/elliottwu/unsup3d) by Shangzhe Wu.
