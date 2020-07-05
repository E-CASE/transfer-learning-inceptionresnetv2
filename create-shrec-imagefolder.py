import os
import os.path
import numpy as np
import shutil

create_data_set = True

data_dir = 'SHREC_14/SHREC14LSSTB_SKETCHES/'

# percent of validation from training image
val_percent = 0

# SHREC-14 classes data folder directories
cls_name = os.listdir(data_dir)
print(cls_name)

# SHREC-14 classes training data folder directories
train_dir = lambda x: data_dir + x + '/train/'
training_directory = [train_dir(x) for x in cls_name]
print(training_directory)

# SHREC-14 classes test data folder directories
test_dir = lambda x: data_dir + x + '/test/'
test_directory = [test_dir(x) for x in cls_name]
print(test_directory)

# number of training data
train_len = 0
for dir in training_directory:
    train_len += len(os.listdir(dir))
print(train_len)

# number of test data
test_len = 0
for dir in test_directory:
    test_len += len(os.listdir(dir))
print(test_len)

shrec_imgfolder = "SHREC14-imagefolder-no-val/"

if create_data_set:
    # create new shrec14 image folder containing a train, validation (formed from training set)
    # and a test split

    if not os.path.exists(shrec_imgfolder):
        os.mkdir(shrec_imgfolder)

    for split in ['train/', 'val/', 'test/']:
        new_folder = os.path.join(shrec_imgfolder, split)
        if not os.path.exists(new_folder):
            os.mkdir(new_folder)

        for dir in cls_name:
            new_cls_folder = os.path.join(shrec_imgfolder, split, dir)
            if not os.path.exists(new_cls_folder):
                os.mkdir(new_cls_folder)

    for dir in cls_name:
        old_path = os.path.join(data_dir, dir, 'train')

        # split training images in old schrec14 sketches folder
        # into validation and training images in new schre14 img folder

        image_list = [images for images in os.listdir(old_path)]
        np.random.shuffle(image_list)
        train_images, val_images = np.split(np.array(image_list), [int(len(image_list) * (1-val_percent))])

        for img_fname in train_images:
            old_pathfolder = os.path.join(old_path, img_fname)
            new_pathfolder = os.path.join(shrec_imgfolder, 'train', dir, img_fname)

            if not os.path.exists(new_pathfolder):
                shutil.copyfile(old_pathfolder, new_pathfolder)

        for img_fname in val_images:
            old_pathfolder = os.path.join(old_path, img_fname)
            new_pathfolder = os.path.join(shrec_imgfolder, 'val', dir, img_fname)

            if not os.path.exists(new_pathfolder):
                shutil.copyfile(old_pathfolder, new_pathfolder)

    for dir in cls_name:
        old_path = os.path.join(data_dir, dir, 'test')
        new_pathfolder = os.path.join(shrec_imgfolder, 'test', dir, img_fname)
        image_list = [images for images in os.listdir(old_path)]

        for img_fname in image_list:
            old_pathfolder = os.path.join(old_path, img_fname)
            new_pathfolder = os.path.join(shrec_imgfolder, 'test', dir, img_fname)
            if not os.path.exists(new_pathfolder):
                shutil.copyfile(old_pathfolder, new_pathfolder)

# check train, val, test numbers
train_num = 0
val_num = 0
test_num = 0

for split in ['train', 'val', 'test']:
    for dir in cls_name:
        folder_path = os.path.join(shrec_imgfolder, split, dir)

        if split == 'train':
            train_num += len(os.listdir(folder_path))

        if split == 'val':
            val_num += len(os.listdir(folder_path))
            
        if split == 'test':
            test_num += len(os.listdir(folder_path))
            
print(train_num, val_num, test_num)