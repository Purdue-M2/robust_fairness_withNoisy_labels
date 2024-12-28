# Robust Fairness Generalization in AI-Generated Face Detection

Li Lin, Zhenhuan Yang, Wenbin Zhang,  Feng Ding, Xin Wang, and Shu Hu*
_________________

This repository is the official implementation of our paper "Robust Fairness Generalization in AI-Generated Face Detection".

## 1. Installation
You can run the following script to configure the necessary environment:

```
cd project_folder
conda create -n RobustFairness python=3.9.0
conda activate RobustFairness
pip install -r requirements.txt
```

## 2. Dataset Preparation

The annotations of FF++, Celeb-DF, DFD, DFDC can be found in [`dataset`](./dataset/). The `intersec_label` annotation (gender and race combined attribute) are obtained from [paper](https://arxiv.org/pdf/2208.05845.pdf). The `Intersection` annotation are obtained from [AI-Face-FairnessBench](https://github.com/Purdue-M2/AI-Face-FairnessBench), which is a cleaner version.

We also shared the processed and corpped images, which can be download the datasets through this [link](https://purdue0-my.sharepoint.com/:f:/g/personal/lin1785_purdue_edu/EtMK0nfxMldAikDxesIo6ckBVHMME1iIV1id_ZsbM9hsqg?e=WayYoy).

Or you can download these datasets from their official website and process them by following the below steps:
- Download [FF++](https://github.com/ondyari/FaceForensics), [Celeb-DF](https://github.com/yuezunli/celeb-deepfakeforensics), [DFD](https://ai.googleblog.com/2019/09/contributing-data-to-deepfake-detection.html) and [DFDC](https://ai.facebook.com/datasets/dfdc/) datasets
- Download annotations for these four datasets according to [paper](https://arxiv.org/pdf/2208.05845.pdf) and their [code](https://github.com/pterhoer/DeepFakeAnnotations), extract the demographics information of all images in each dataset. 
- Extract, align and crop face using [DLib](https://www.jmlr.org/papers/volume10/king09a/king09a.pdf), and save them to `/path/to/cropped_images/`
- Split cropped images in each dataset to train/val/test with a ratio of 60%/20%/20% without identity overlap.

The csv file we provide in [`dataset`](./dataset/) are formatted as:
  
| Column                     | Description                                                         |
|----------------------------|---------------------------------------------------------------------|
| Image Path                 | Path to the image file                                              |
| Ground Truth Gender        | Gender label: 1 - Male, 0 - Female                                  |
| Ground Truth Age           | Age label: 0 - Young, 1 - Middle-aged, 2 - Senior, 3 - Others       |
| Ground Truth Race          | Race label: 0 - Asian, 1 - White, 2 - Black, 3 - Others             |
| Intersection (cleaner)               |  0-(Male,Asian), 1-(Male,White), 2-(Male,Black), 3-(Male,Others), 4-(Female,Asian), 5-(Female,White), 6-(Female,Black), 7-(Female,Others)|
| Target                     | Label indicating real (0) or fake (1) image                         |
| Specific                   | Manipulation method in FF++: 1-Deepfakes, 2-Face2Face, 3-FaceSwap, 4-NeuralTextures, 5-FaceShifter   |
| intersec_label (noisy)              |  0-(Male,Asian), 1-(Male,White), 2-(Male,Black), 3-(Male,Others), 4-(Female,Asian), 5-(Female,White), 6-(Female,Black), 7-(Female,Others)|

## 3. Load Pretrained Weights
Before running the training code, make sure you load the pre-trained weights. You can download *Xception* model trained on ImageNet (through this [link](http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth)) and put it under [`./training/pretrained`](./training/pretrained).

## 4. Train
To run the training code, you should first go to the [`./training/`](./training/) folder. You can train  **PG-FDD** and **Robust PG-FDD** proposed in our work by excuting below command:

### Train **PG-FDD**

```
cd training

python train_pgfdd.py
```

### Train **Robust PG-FDD**
```
cd training

python train_robust_pgfdd.py
```

You can adjust the parameters in [`train_pgfdd.py`](training/train_pgfdd.py) or [`train_robust_pgfdd.py`](training/train_robust_pgfdd.py) to specify the parameters, *e.g.,* training dataset, batchsize, learnig rate, *etc*.

`--lr`: learning rate, default is 0.0005. 

`--checkpoints`: folder to save model checkpoints.

` --datapath`: /path/to/train_dataset_folder, default='../dataset/ff++/'.

` --test_datapath`: /path/to/test_dataset_file, default='../dataset/ff++/test_noise.csv'.

`--train_batchsize`: train batch size, default is 16.

`--model`: detector name: 'robust_pgfdd'.

## 5. Test
* For model testing, we provide a python file to test our model by running `python test.py`. 

	`--inter_attribute`: intersectional group names divided by '-': male,asian-male,white-male,black-male,others-nonmale,asian-nonmale,white-nonmale,black-nonmale,others-young-middle-senior-ageothers

	`--checkpoints`: /path/to/saved/model_folder.

	`--specific_checkpoint`: /path/to/saved/model.pth. (specific checkpoint file to test)

	`--savepath`: /where/to/save/predictions.npy(labels.npy)/results/ 

	`--model_structure`: detector name: robust_pgfdd.

	`--batch_size`: testing batch size: default is 32.

* After testing, for metric calculation, we provide `python fairness_metrics.py` to print all the metrics. It is automatically called in `test.py`.

#### 📝 Note
Change `--inter_attribute` and `--single_attribute` for different testing dataset:

```
### ff++, dfdc
--inter_attribute male,asian-male,white-male,black-male,others-nonmale,asian-nonmale,white-nonmale,black-nonmale,others-young-middle-senior-ageothers \


### celebdf, dfd
--inter_attribute male,white-male,black-male,others-nonmale,white-nonmale,black-nonmale,others-young-middle-senior-ageothers \

```

## 📦 Provided Backbones
|                  | File name                               | Paper                                                                                                                                                                                                                                                                                                                                                         |
|------------------|-----------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Xception          | [xception.py](./training/networks/xception.py)         | [Xception: Deep learning with depthwise separable convolutions](https://openaccess.thecvf.com/content_cvpr_2017/html/Chollet_Xception_Deep_Learning_CVPR_2017_paper.html) |
| ResNet50          | [resnet50.py](training/networks/resnet50.py)       | [Deep Residual Learning for Image Recognition](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)                                                                                                                                                                                                                                                                                              |
| EfficientNet-B3      | [efficientnetb3.py](./training/networks/efficientnetb3.py) | [Efficientnet: Rethinking model scaling for convolutional neural networks](http://proceedings.mlr.press/v97/tan19a.html)                                                                                                                                                                                                                  |
| EfficientNet-B4      | [efficientnetb4.py](./training/networks/efficientnetb4.py) | [Efficientnet: Rethinking model scaling for convolutional neural networks](http://proceedings.mlr.press/v97/tan19a.html) 
