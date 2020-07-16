# Video Prediction via Example Guidance (VPEG)

Project Page: https://sites.google.com/view/vpeg-supp/home

Code Release for "Video Prediction via Example Guidance" (ICML 2020), under construction.

## On the RobotPush Dataset:

### 1. Download the dataset: 

http://rail.eecs.berkeley.edu/datasets/bair_robot_pushing_dataset_v0.tar

### 2. Extract the data: 

python extract.py

### 3. Train the feature extractor: 

python train_vpeg_feature.py --dataset bair --model vgg --g_dim 128 --z_dim 16 --beta 0.0001 --n_past 5 --n_future 10 --channels 3 --log_dir logs/bair-match/ --data_dir /your/data/path

### 4. Train the motion predictor: 

python train_vpeg.py --dataset bair-match --model vgg --g_dim 128 --z_dim 16 --beta 0.0001 --alpha 0.01 --n_past 5 --n_future 10 --channels 3 --log_dir logs/bair-match/ --data_dir /your/data/path --feat_model_dir /path/to/trained/model/at/3rd/step

### 5. Generate the predicted sequences:

python generate_vpeg.py --dataset bair-match --model vgg --g_dim 128 --z_dim 16 --beta 0.0001 --alpha 0.01 --n_past 5 --n_future 25 --n_eval 30 --channels 3 --log_dir logs/bair-match/ --data_dir /your/data/path --feat_model_dir /path/to/trained/model/at/4th/step

The general idea of our work is very simple and straightforward. On this dataset our code is built based on [this repo](https://github.com/edenton/svg). The major contribution of our work is reflected in the [train_vpeg.py](https://github.com/xjwxjw/VPEG/blob/master/RobotPush/train_vpeg.py) (from L.355-L.390) and train_vpeg_feature.py. Please refer to the comment in the code for more details (With indicator ""Our work"").

## On the PennAction Dataset:

### 1. Follow [this repo](https://github.com/YunjiKim/Unsupervised-Keypoint-Learning-for-Guiding-Class-conditional-Video-Prediction) to download/preprocess the PennAction data. 

### 2. Follow [this repo](https://github.com/YunjiKim/Unsupervised-Keypoint-Learning-for-Guiding-Class-conditional-Video-Prediction) to generate pseudo-keypoints labels.

### 3. Train the motion predictor:

python train.py --mode motion_generator --config configs/penn.yaml

### 4. Generate the predicted seuqences:

python evaluate.py --config configs/penn.yaml --checkpoint_stage1 ./PretrainedModel/stage1/model.ckpt --checkpoint_stage2 ./results_VPEG/motion_generator/model.ckpt-60000 --save_dir ./gif_VPEG

On this dataset our code is built based on [this repo](https://github.com/YunjiKim/Unsupervised-Keypoint-Learning-for-Guiding-Class-conditional-Video-Prediction). The major contribution of our work on this dataset is reflected in the [models/motion_generator_model.py](https://github.com/xjwxjw/VPEG/blob/master/PennAction/models/motion_generator_model.py) (from L.179-L.234 and from L.347-L.384). In the config file (configs/penn.yaml), setting the variable ''sth_pro'' as True enables to train the baseline model with our method. Please refer to the comment in the code for more details (With indicator ""Our work"").
