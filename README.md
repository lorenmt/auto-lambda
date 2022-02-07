# Auto-Lambda
This repository contains the source code of Auto-Lambda and baselines from the paper, "Auto-Lambda: Disentangling Dynamic Task Relationships". See more discussion in our [project page](https://shikun.io/projects/auto-lambda). 

## Multi-task Methods
We cover all weighting and gradient-based baselines presented in the paper for computer vision tasks: Dense Prediction Tasks (for NYUv2 and CityScapes) and Multi-domain Classification Tasks (CIFAR-100). 

Specifically, we cover the implementation of these following multi-task optimisation methods:

### Weighting-based:
- **Equal** - All task weightings are 1. `-weight equal`
- **Uncertainty** - [https://arxiv.org/abs/1705.07115](https://arxiv.org/abs/1705.07115) `-weight uncert`
- **Dynamic Weight Average** - [https://arxiv.org/abs/1803.10704](https://arxiv.org/abs/1803.10704) `-weight dwa`
- **Auto-Lambda** - Our Proposed Method. `-weight autol`

### Gradient-based
- **GradDrop** -  [https://arxiv.org/abs/2010.06808](https://arxiv.org/abs/2010.06808) `-grad_method graddrop`
- **PCGrad** - [https://arxiv.org/abs/2001.06782](https://arxiv.org/abs/2001.06782) `-grad_method pcgrad`
- **CAGrad** - [https://arxiv.org/abs/2110.14048](https://arxiv.org/abs/2110.14048) `-grad_method cagrad`

*Note: Applying a combination of both weighting and gradient-based methods can further improve performance.*

## Datasets
We follow the same data pre-processing format in my previous project: [MTAN](https://github.com/lorenmt/mtan) which experimented on:

- [**NYUv2 [3 Tasks]**](https://www.dropbox.com/sh/86nssgwm6hm3vkb/AACrnUQ4GxpdrBbLjb6n-mWNa?dl=0)  - 13 Class Segmentation + Depth Estimation + Surface Normal.
- [**CityScapes [3 Tasks]**](https://www.dropbox.com/sh/qk3cr18d55d08gj/AAA5OCTPNFDEDk5fZsmCfmrAa?dl=0) - 19 Class Segmentation + 10 Class Part Segmentation + Disparity (Inverse Depth) Estimation.

*Note: We have included a new task: [Part Segmentation](https://github.com/pmeletis/panoptic_parts) for CityScapes dataset. The pre-processing file for CityScapes is also included in the `dataset` folder.*


## Experiments
All experiments can be trained with different flags (hyper-parameters) when running each training script. We briefly introduce some important flags below.

| Flag Name     | Usage                                                                                                                                                 | Comments                                                                            |
|---------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| `network`     | choose multi-task network: `split, mtan`                                                                                                              | both architectures are based on ResNet-50, only available in dense prediction tasks |
| `dataset`     | choose dataset: `nyuv2, cityscapes`                                                                                                                   | only available in dense prediction tasks                                            |
| `weight`      | choose weighting method: `equal, uncert, dwa, autol`                                                                                                  | only `autol` will behave differently when set to different primary tasks            |
| `grad_method` | choose gradient method: `graddrop, pcgrad, cagrad`                                                                                                    | `weight` and `grad_method` can be applied together                                  |
| `task`        | choose primary tasks: `seg, depth, normal` for NYUv2, `seg, part_seg, disp` for CityScapes, `all`: a combination of standard 3 tasks as primary tasks | only available in dense prediction tasks                                            |
| `with_noise`  | toggle on to add noise prediction task for training (to evaluate robustness in auxiliary learning setting)                                            | only available in dense prediction tasks                                            |
| `subset_id`   | choose domain ID for CIFAR-100, choose `-1` for multi-task learning setting                                                                           | only available in CIFAR-100 tasks                                                   |
| `autol_init`  | initialisation of Auto-Lambda, default `0.1`                                                                                                          | only available when applied Auto-lambda as weighting method                         |
| `autol_lr`    | learning rate of Auto-Lambda, default `1e-4`  for NYUv2 and `3e-5` for CityScapes                                                                     | only available when applied Auto-lambda as weighting method                         |

Training Auto-Lambda in Multi-task / Auxiliary Learning Mode:
```
python trainer_dense.py --dataset [nyuv2, cityscapes] --task [PRIMARY_TASK] --weight autol --gpu 0   # for NYUv2 or CityScapes dataset
python trainer_cifar.py --subset_id [PRIMARY_DOMAIN_ID] --weight autol --gpu 0   # for CIFAR-100 dataset
```

Training in Single-task Learning Mode:
```
python trainer_dense_single.py --dataset [nyuv2, cityscapes] --task [PRIMARY_TASK]  --gpu 0   # for NYUv2 or CityScapes dataset
python trainer_cifar_single.py --subset_id [PRIMARY_DOMAIN_ID] --gpu 0   # for CIFAR-100 dataset
```

## Benchmark
For standard 3 tasks in NYUv2 (without dense prediction task) in multi-task learning setting, you should expect the following results.

| Method               | Sem. Seg. (mIOU) | Depth (aErr.) | Normal (mDist.) | Delta MTL - |
|----------------------|------------------|---------------|-----------------|-------------|
| Equal	               | 44.64	           | 43.32	        | 24.48	          | +3.57%      |
| DWA	                 | 45.14            | 	43.06        | 	24.17          | 	+4.58%     |
| GradDrop             | 45.39            | 43.23         | 24.18           | +4.65%      |
| PCGrad               | 45.15            | 42.38         | 24.13           | +5.09%      |
| Uncertainty          | 	45.98           | 	41.26        | 	24.09          | 	+6.50%     |
| CAGrad               | 46.14            | 41.91         | 23.52           | +7.05%      |
| Auto-Lambda          | 	47.17           | 	40.97	       | 23.68           | 	+8.21%     |
| Auto-Lambda + CAGrad | 	48.26           | 	39.82	       | 22.81           | 	+11.07%    |


## Citation
If you found this code/work to be useful in your own research, please considering citing the following:

```
@article{liu2022auto-lambda,
  title={Auto-Lambda: Disentangling Dynamic Task Relationships},
  author={Liu, Shikun and James, Stephen and Davison, Andrew J and Johns, Edward},
  journal={arXiv preprint},
  year={2022}
}
```

## Acknowledgement
We would like to thank [@Cranial-XIX](https://github.com/Cranial-XIX) for his amazing clean implementation for gradient-based optimisation methods.

## Contact
If you have any questions, please contact `sk.lorenmt@gmail.com`.