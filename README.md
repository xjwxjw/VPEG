# Video Prediction via Example Guidance (VPEG)

Project Page: https://sites.google.com/view/vpeg-supp/home

Code Release for "Video Prediction via Example Guidance" (ICML 2020), under construction.

On the RobotPush Dataset:

1. Download the dataset: http://rail.eecs.berkeley.edu/datasets/bair_robot_pushing_dataset_v0.tar

2. Extract the data: python extract.py

3. Train the feature extractor: 

python train_vpeg_feature.py --dataset bair --model vgg --g_dim 128 --z_dim 16 --beta 0.0001 --n_past 5 --n_future 10 --channels 3 --log_dir logs/bair-match/ --data_dir /your/data/path

4. Train the motion predictor: 

python train_vpeg.py --dataset bair-match --model vgg --g_dim 128 --z_dim 16 --beta 0.0001 --alpha 0.01 --n_past 5 --n_future 10 --channels 3 --log_dir logs/bair-match/ --data_dir /your/data/path --feat_model_dir /path/to/trained/model/at/previous/step

The general idea behind our work is very easy to implement. On this dataset our code is built based on this repo (https://github.com/edenton/svg). The major contribution of our work is reflected in the train_vpeg.py (from L.355-L.390) and train_vpeg_feature.py. Please refer to the comment in the code for more details.
