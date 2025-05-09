# Official Implementation for EBPR
([https://arxiv.org/pdf/2205.14566.pdf](https://www.sciencedirect.com/science/article/abs/pii/S0167865525001370))

### Framework:  

1. train on the source domain;
2. train on target dataset.

### Prerequisites:
- python == 3.6.8
- pytorch ==1.1.0
- torchvision == 0.3.0
- numpy, scipy, sklearn, ot, argparse

### Dataset:

- Please manually download the datasets [Office](https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view), [Office-Home](https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view), [VisDA-C](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification) from the official websites.


### Training:
##### Office-31 dataset
	```python
    # train source model
    python train_source.py --dset office --s 0 --max_epoch 50
    # train target model
    python train_target.py --dset office --emt 0.999 --output test --gpu_id 7 --s 0 --t 1
    python train_target.py --dset office --emt 0.999 --output test --gpu_id 7 --s 0 --t 2
	```


### Citation

If you find this code useful for your research, please cite our papers
```
@article{MENG2025,
	title = {Energy-based pseudo-label refining for source-free domain adaptation},
	journal = {Pattern Recognition Letters},
	year = {2025},
	issn = {0167-8655},
	doi = {https://doi.org/10.1016/j.patrec.2025.04.004},
	url = {https://www.sciencedirect.com/science/article/pii/S0167865525001370}
}
