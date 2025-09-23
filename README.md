# EGA-Ploc

In this work, we propose EGA-Ploc, an efficient deep learning tool that integrates a novel linear attention mechanism for efficiently and effectively acquiring discriminative representations from IHC images, a hierarchical multi-scale architecture  to preserve both fine-grained subcellular patterns and global spatial context, and an enhanced  multi-label objective function to counteract dataset imbalance. By jointly optimizing representation learning and class distribution modeling, EGA-Ploc overcomes the limitations of existing patch-based or downsampling-reliant approaches.

This repository contains the official code for our paper: [EGA-Ploc: An Efficient Global-Local Attention Model for Multi-label Protein Subcellular Localization Prediction on the Immunohistochemistry Images](https://ieeexplore.ieee.org/document/11175487)

+ Sep 2025: This work is accepted by IEEE Journal of Biomedical and Health Informatics.😃
+ Sep 2025: We will release training code in the middle of October.

Online demo is available now🎉🎉🎉 Anyone can quickly experience our model by visiting this link: <https://huggingface.co/spaces/austinx25/EGA-Ploc>

## 1. Platform and Dependency

### 1.1 Platform

* Ubuntu 9.4.0-1ubuntu1~20.04.1

* NVIDIA® V100 Tensor Core GPU(32GB) * 8
  
  ### 1.2 Dependency
  
  | Requirements      | Release  |
  | ----------------- | -------- |
  | CUDA              | 11.3     |
  | Python            | 3.8.15   |
  | pytorch           | 1.11.0   |
  | torchvision       | 0.12.0   |
  | torchaudio        | 0.11.0   |
  | cudatoolkit       | 11.3     |
  | pandas            | 1.2.4    |
  | fvcore            | 0.1.5    |
  | opencv-python     | 4.6.0.66 |
  | timm              | 0.6.12   |
  | scipy             | 1.9.3    |
  | einops            | 0.6.0    |
  | matplotlib        | 3.5.1    |
  | scikit-learn      | 1.1.2    |
  | tensorboard       | 2.11.0   |
  | adabelief-pytorch | 0.2.0    |
  
  ## 2. Project Catalog Structure
  
  ### 2.1 data
  
  > Download and save the data annotation information to this folder. Such files generally have a .csv extension
  > Vislocas data annotation files has been deposited at Zenodo <https://doi.org/10.5281/zenodo.10632698>
  > HPA18 data has been deposited at <http://www.csbio.sjtu.edu.cn/bioinf/GraphLoc>

The following files should be placed directly in the `data` directory.  

| File            | Descriptrion                      |
| --------------- | --------------------------------- |
| data_train.csv  | The full training set of Vislocas |
| HPA18_train.csv | The full training set of HPA18    |
| HPA18_train.csv | The full training set of HPA18    |
| HPA18_test.csv  | The test set of HPA18             |

### 2.2 dataset

> This folder should contains the source files for IHC images 

### 2.3 datasets

> This folder stores the code files for data loading.

* ihc.py
  
  > This file includes Vislocas data loading code.

* HPA18.py
  
  > This file includes HPA18 data loading code.

* build.py
  
  > This file includes building dataset code.

* loader.py
  
  > This file includes constructing loader code.

### 2.4 assets

- This folders stores 12 IHC images for single image test used in `tools\test_demo.py`.

- The title of the subfolder is the label for the images in the folder

### 2.5 logs

> This folder is used to store the output log messages.

### 2.6 models

> This folder stores model-related code files, including Visloacas model code, loss function code, and model training-related code.

* EGA-Ploc
  
  > This file includes EGA-Ploc model code.

* classifier_model.py
  
  > This file includes load model code.

* train_classifier.py
  
  > This file includes model training-related code.

* loss.py
  
  > This file includes loss function code.

* criterion.py
  
  > This file includes criterion-related code.

### 2.7 results

> This folder is used to store the output models and prediction results.

### 2.8 tools

> This folder stores code files for model prediction. Once our article is accepted, we will publish the training code

* test.py
  
  > This file includes model testing code. Depending on the user input parameters, the performance of EGA-Ploc on Vislocas and HPA18 datasets can be tested separately

* test_demo.py
  
  > This file contains the code for EGA-Ploc to test a single IHC image. To run the code in this file, run test_demo.sh in the project root directory


### 2.9 utils

> This folder stores the optimiser, scheduler, checkpoint and other utilities code files.

* args.py
  
  > This file includes parameters that can be customized and adjusted before running the .py file

* checkpoint.py
  
  > This file includes checkpoint code.

* config_defaults.py
  
  > This file includes the parameter configuration code.

* distributed.py
  
  > This file includes the code for distributed training.

* eval_metrics.py
  
  > This file includes the utilities code for calculating the performance metrics.

* optimizer.py
  
  > This file includes the optimizer code.

* scheduler.py
  
  > This file includes the scheduler code.

## 3. How to Test

### 3.1 Check the data annotation file in the data folder

There should exist four .csv file in `data` folder mentioned in Section 2.1

- If the file starting with `data_` is missing, please download it at <https://doi.org/10.5281/zenodo.10632698>
- If the file starting with `HPA18_` is missing, please contact the corresponding author's email address for access: `kooyang@aliyun.com`

### 3.2 Download the images from HPA.

You can get the original image needed for this file in two ways:

1. (Recommand) Download from < https://pan.baidu.com/s/1W8LA4P8uIW2SKcZ3sJuQaw>, the password is `rkjb`. **"IHC"** folder includes the image of Vislocas dataset, and **"HPA18"** folder includes the image of HPA18 dataset. If the website does not work, you can contact the corresponding author for the original image.
2. Download the original image via the `URL` column within the data annotation file

### 3.3 Download the checkpoint (Options)
if you want to test the model, you should download checkpoint file from <https://jxstnueducn-my.sharepoint.com/:f:/g/personal/wanboyang_jxstnu_edu_cn/EpEDB3GcXMZFvRz9lQaBHswBYTEWUDF6ThPBHWqEPB-eUQ?e=jsSoY0> (<https://pan.baidu.com/s/1nxflmaSpQkhEF0src_Tt3w> specially for Chinese researcher, the password is `pusd`) **AIPLoc_fa_4_cl1_3000_wd-005_best_model.pth** represents the best weight trained from Vislocas dataset and **AIPLoc_fa_4_dcl1_3000_wd-005_mlce_best_model.pth** denotes the best weight trained from HPA-18 dataset.

1. put **AIPLoc_fa_4_cl1_3000_wd-005_best_model.pth** into "results -> IHC -> AIPLoc_fa_4_cl1_3000_wd-005_mlce" path and rename the checkpoint file to "best_model.pth".
2. put **AIPLoc_fa_4_dcl1_3000_wd-005_mlce_best_model.pth** into "results -> HPA18 -> AIPLoc_discount_fa_4_cl1_3000_wd-005_mlce" path and rename the checkpoint file to "best_model.pth".

### 3.4 Install the environment.

Before you start, we recommend that you install the python environment management tool `Anaconda`. Afterwards, go to the project root environment in the console interface and execute the command  `conda env create -f environment.yml`

### 3.5 Start to Test

- When you need to test the entire dataset, run the `test.sh` file in the root directory.
- When you need to quickly test the effectiveness of the model predictions, you can run `test_demo.sh` to get test results for a single image. In this step, you can test the effect of EGA-Ploc trained on different datasets by adjusting the following parameters:

| Parameters        | default                              | Descriptrion                                                                      |
| ----------------- | ------------------------------------ | --------------------------------------------------------------------------------- |
| dataset           | IHC                                  | determines which dataset profile will be used                                     |
| single_image_path | ./assets/Cytopl;Mito/55449_A_1_2.jpg | determines the relative path of the test image, only takes effect in test_demo.py |
