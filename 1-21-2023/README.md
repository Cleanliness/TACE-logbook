# Jan 21 2023
Finalizing PNET transfer learning + bootstrapping experiments.

## Overview
- Run finetuning on pretrained PNET dataset

## Experiment 1: PNET Transfer learning
Task 606: Model using TACE plans, trained on PNET dataset
- 60-20-20 Train-Val-Test split

### Procedure
1. Preprocess TACE dataset (done) `nnUNet_plan_and_preprocess -t <tacenum>`
2. Preprocess PNET dataset using plans from TACE (done) `nnUNet_plan_and_preprocess -t <pnetnum> -overwrite_plans <tace_plans> -overwrite_plans_identifier <tacenum>`
3. Train U-Net using plans from TACE dataset (done) `nnUNet_train 3d_fullres nnUNetTrainerV2 <pnet_taskname> all -p <pretrained_plan_name>`
4. Run fine tuning on TACE dataset `nnUNet_train 3d_fullres nnUNetTrainerV2 <pnet_taskname> -pretrained_weights PATH_TO_MODEL_FINAL_CHECKPOINT`

### Results
