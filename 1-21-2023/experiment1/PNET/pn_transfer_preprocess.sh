#!/bin/bash
#SBATCH --account=def-amartel
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

module load python/3.8
source $HOME/tace_new/bin/activate
export nnUNet_raw_data_base=/home/royga/projects/def-amartel/royga/nn_unet_test/nnUNet_raw_data_base
export nnUNet_preprocessed=/home/royga/projects/def-amartel/royga/nn_unet_test/nnUNet_preprocessed
export RESULTS_FOLDER=/home/royga/projects/def-amartel/royga/nn_unet_test/results
nnUNet_plan_and_preprocess -t 606 -pl3d ExperimentPlanner3D_v21_Pretrained -overwrite_plans /home/royga/projects/def-amartel/royga/nn_unet_test/nnUNet_preprocessed/Task604_TACE_stitch/nnUNetPlansv2.1_plans_3D.pkl -overwrite_plans_identifier PNET_TRANSFER_TEST
