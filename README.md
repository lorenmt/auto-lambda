# Auto-Lambda
This repository contains the source code of Auto-Lambda and baselines from the paper, [Auto-Lambda: Disentangling Dynamic Task Relationships](https://arxiv.org/abs/2202.03091). 

We encourage readers to check out our [project page](https://shikun.io/projects/auto-lambda), including more interesting discussions and insights which are not covered in our technical paper.

## Multi-task Methods
We implemented all weighting and gradient-based baselines presented in the paper for computer vision tasks: Dense Prediction Tasks (for NYUv2 and CityScapes) and Multi-domain Classification Tasks (for CIFAR-100). 

Specifically, we have covered the implementation of these following multi-task optimisation methods:

### Weighting-based:
- **Equal** - All task weightings are 1. `--weight equal`
- **Uncertainty** - [https://arxiv.org/abs/1705.07115](https://arxiv.org/abs/1705.07115) `--weight uncert`
- **Dynamic Weight Average** - [https://arxiv.org/abs/1803.10704](https://arxiv.org/abs/1803.10704) `--weight dwa`
- **Auto-Lambda** - Our approach. `--weight autol`

### Gradient-based:
- **GradDrop** -  [https://arxiv.org/abs/2010.06808](https://arxiv.org/abs/2010.06808) `--grad_method graddrop`
- **PCGrad** - [https://arxiv.org/abs/2001.06782](https://arxiv.org/abs/2001.06782) `--grad_method pcgrad`
- **CAGrad** - [https://arxiv.org/abs/2110.14048](https://arxiv.org/abs/2110.14048) `--grad_method cagrad`

*Note: Applying a combination of both weighting and gradient-based methods can further improve performance.*

## Datasets
We applied the same data pre-processing following our previous project: [MTAN](https://github.com/lorenmt/mtan) which experimented on:

- [**NYUv2 [3 Tasks]**](https://www.dropbox.com/sh/86nssgwm6hm3vkb/AACrnUQ4GxpdrBbLjb6n-mWNa?dl=0)  - 13 Class Segmentation + Depth Estimation + Surface Normal. [288 x 384] Resolution.
- [**CityScapes [3 Tasks]**](https://www.dropbox.com/sh/qk3cr18d55d08gj/AAA5OCTPNFDEDk5fZsmCfmrAa?dl=0) - 19 Class Segmentation + 10 Class Part Segmentation + Disparity (Inverse Depth) Estimation. [256 x 512] Resolution.

*Note: We have included a new task: [Part Segmentation](https://github.com/pmeletis/panoptic_parts) for CityScapes dataset. The pre-processing file for CityScapes has also been included in the `dataset` folder.*


## Experiments
All experiments were written in `PyTorch 1.7` and can be trained with different flags (hyper-parameters) when running each training script. We briefly introduce some important flags below. 

| Flag Name     | Usage                                                                                                                                    | Comments                                                                            |
|---------------|------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| `network`     | choose multi-task network: `split, mtan`                                                                                                 | both architectures are based on ResNet-50; only available in dense prediction tasks |
| `dataset`     | choose dataset: `nyuv2, cityscapes`                                                                                                      | only available in dense prediction tasks                                            |
| `weight`      | choose weighting-based method: `equal, uncert, dwa, autol`                                                                               | only `autol` will behave differently when set to different primary tasks            |
| `grad_method` | choose gradient-based method: `graddrop, pcgrad, cagrad`                                                                                 | `weight` and `grad_method` can be applied together                                  |
| `task`        | choose primary tasks: `seg, depth, normal` for NYUv2, `seg, part_seg, disp` for CityScapes, `all`: a combination of all standard 3 tasks | only available in dense prediction tasks                                            |
| `with_noise`  | toggle on to add noise prediction task for training (to evaluate robustness in auxiliary learning setting)                               | only available in dense prediction tasks                                            |
| `subset_id`   | choose domain ID for CIFAR-100, choose `-1` for the multi-task learning setting                                                          | only available in CIFAR-100 tasks                                                   |
| `autol_init`  | initialisation of Auto-Lambda, default `0.1`                                                                                             | only available when applying Auto-Lambda                        |
| `autol_lr`    | learning rate of Auto-Lambda, default `1e-4`  for NYUv2 and `3e-5` for CityScapes                                                        | only available when applying Auto-Lambda                       |

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

*Note: All experiments in the original paper were trained from scratch without pre-training.*

## Benchmark
For standard 3 tasks in NYUv2 (without dense prediction task) in the multi-task learning setting with Split architecture, please follow the results below.

| Method               | Type | Sem. Seg. (mIOU) | Depth (aErr.) | Normal (mDist.) | Delta MTL |
|----------------------|-------|-----------|---------------|-----------------|-----------|
|Single | - | 43.37	| 52.24	        |22.40| - |
| Equal	               | W |44.64	           | 43.32	        | 24.48	          | +3.57%    |
| DWA	                 | W |45.14            | 	43.06        | 	24.17          | 	+4.58%   |
| GradDrop             | G |45.39            | 43.23         | 24.18           | +4.65%    |
| PCGrad               | G | 45.15            | 42.38         | 24.13           | +5.09%    |
| Uncertainty          | W |	45.98           | 	41.26        | 	24.09          | 	+6.50%   |
| CAGrad               | G |46.14            | 41.91         | 23.52           | +7.05%    |
| Auto-Lambda          | W |	47.17           | 	40.97	       | 23.68           | 	+8.21%   |
| Auto-Lambda + CAGrad | W + G|	48.26           | 	39.82	       | 22.81           | 	+11.07%  |

*Note: The results were averaged across three random seeds. You should expect the error range less than +/-1%.*

## Citation
If you found this code/work to be useful in your own research, please considering citing the following:

```
@article{liu2022auto-lambda,
  title={Auto-Lambda: Disentangling Dynamic Task Relationships},
  author={Liu, Shikun and James, Stephen and Davison, Andrew J and Johns, Edward},
  journal={arXiv preprint arXiv:2202.03091},
  year={2022}
}
```

## Acknowledgement
We would like to thank [@Cranial-XIX](https://github.com/Cranial-XIX) for his clean implementation for gradient-based optimisation methods.

## Contact
If you have any questions, please contact `sk.lorenmt@gmail.com`.
