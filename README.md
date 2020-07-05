## SHREC14 Transfer Learn InceptionResnetV2 for sketch recognition

Transfer learning script for a pre-trained inceptionresnetv2 from [pretrained-models.pytorch](https://github.com/Cadene/pretrained-models.pytorch), for sketch recognition. Achieved a SHREC14 sketch classification test accuracy of 80.18%, on Pytorch 1.5.1.

Uses a label smoothed cross-entropy loss. 

# Model
Download pre-trained weights for inceptionresnetv2.py [here](https://drive.google.com/file/d/1JxG5_qW7Z3owe-xwX3uj71x57vUJCSrf/view?usp=sharing) and store in script directory. 

# Dataset

Uses SHREC14 sketches transformed into an imagefolder structure with no validation set, download SHREC14-imagefolder-no-val [here](https://drive.google.com/file/d/11EEXDuGaX0Rendz3453p9sEdBx758J5e/view?usp=sharing) and store in script directory. 

To create your own SHREC14 imagefolder download SHREC14 dataset [here](https://drive.google.com/file/d/1XA53BG3oxPvx_0SuipnJjyr_GAJas0lu/view?usp=sharing) and use the script create-shrec-imagefolder.py to customise train, validation, and test split.

# Run
configuration used to obtain 80.18% accuracy,

python transfer-learn-script.py --batchSize=8 --learning_rate=4e-5 --train_epochs=20 --steplar_stepsize=9 --steplr_gamma=0.1 --smoothing=0.2


