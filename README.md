## Pixel_DiffusionSeg:Instance segmentation based on multilevel ddeformable self-attention and diffusion models



**Pixel_DiffusionSeg is a method that uses multi-level deformable self-attention and pixel decoders to improve the accuracy of instance segmentation using diffusion models.**
![](Pixel_DiffusionSeg.jpg)
This method can better preserve the details of the image,We hope our work could serve as a simple yet effective baseline, which could inspire designing more efficient diffusion frameworks for challenging discriminative tasks.


> **Pixel_DiffusionSeg:Instance segmentation based on multilevel ddeformable self-attention and diffusion models**             
> Hongqian Chen,Rui Yang


## Getting Started
### Installation

The codebases are built on top of [Detectron2](https://github.com/facebookresearch/detectron2), 
[Sparse R-CNN](https://github.com/PeizeSun/SparseR-CNN), 
and [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch).
Thanks very much.

#### Requirements
- Linux or macOS with Python ≥ 3.6
- PyTorch ≥ 1.9.0 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  You can install them together at [pytorch.org](https://pytorch.org) to make sure of this
- OpenCV is optional and needed by demo and visualization

#### Steps
1. Install Detectron2 following https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md#installation.

2. Prepare datasets
```
mkdir -p datasets/coco

ln -s /path_to_coco_dataset/annotations datasets/coco/annotations
ln -s /path_to_coco_dataset/train2017 datasets/coco/train2017
ln -s /path_to_coco_dataset/val2017 datasets/coco/val2017
```
3. Prepare pretrain models

Pixel_DiffusionSeg uses the ResNet-50 backbone, and pre-trained ResNet-50 models can be downloaded automatically with Detectron2.

### Trained Pixel_DiffusionSeg

```
python train_net.py --num-gpus 1 --config-file configs/coco.res50.yaml
```

Command Line Args: Namespace(config_file='configs/coco.res50.yaml', dist_url='tcp://127.0.0.1:49153', eval_only=False, machine_rank=0 , num_gpus=1, num_machines=1, opts=[], resume=False)

### Evaluate Pixel_DiffusionSeg
```
python train_net.py --num-gpus 1 \
    --config-file configs/coco.res50.yaml \
    --eval-only MODEL.WEIGHTS path/to/model.pth
```
### Inference Demo with Pre-trained Models
We provide a command line tool to run a simple demo following [Detectron2](https://github.com/facebookresearch/detectron2/tree/main/demo#detectron2-demo).

```bash
python demo.py --config-file configs/coco.res50.yaml \
    --input image.jpg --opts MODEL.WEIGHTS Pixel_DiffusionSeg_coco_res50.pth
```
We need to specify MODEL.WEIGHTS to a model This command will run the inference and show visualizations in an OpenCV window.

For details of the command line arguments, see `demo.py -h` or look at its source code
to understand its behavior. Some common arguments are:
* To run __on your webcam__, replace `--input files` with `--webcam`.
* To run __on a video__, replace `--input files` with `--video-input video.mp4`.
* To run __on cpu__, add `MODEL.DEVICE cpu` after `--opts`.
* To save outputs to a directory (for images) or a file (for webcam or video), use `--output`.


## Acknowledgement
Many thanks to the nice work of DiffusionInst @[chenhaoxing](https://github.com/chenhaoxing). Our codes and configs follow [DiffusionInst](https://github.com/chenhaoxing/DiffusionInst).


## Contacts
Please feel free to contact us if you have any problems.

Email: [chenhq@th.btbu.edu.cn](chenhq@th.btbu.edu.cn) 