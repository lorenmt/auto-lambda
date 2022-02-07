import glob
import os
from PIL import Image

root = 'dataset'

im_train_root = root + 'cityscapes/train/image'
im_val_root = root + '/cityscapes/val/image'
label_train_root = root + '/cityscapes/train/seg'
label_val_root = root + '/cityscapes/val/seg'
part_train_root = root + '/cityscapes/train/part_seg'
part_val_root = root + '/cityscapes/val/part_seg'
depth_train_root = root + '/cityscapes/train/depth'
depth_val_root = root + '/cityscapes/val/depth'

os.makedirs(im_train_root)
os.makedirs(im_val_root)
os.makedirs(label_train_root)
os.makedirs(label_val_root)
os.makedirs(part_train_root)
os.makedirs(part_val_root)
os.makedirs(depth_train_root)
os.makedirs(depth_val_root)


# Images
train_im_list = glob.glob(root + '/leftImg8bit_trainvaltest/leftImg8bit/train/*')
counter = 0
for city in train_im_list:
    im_list = glob.glob(city + '/*.png')
    im_list.sort()
    for i in im_list:
        im = Image.open(i)
        im = im.resize((512, 256))
        im.save(im_train_root + '/{}.png'.format(counter))
        counter += 1
print('Training RGB images processing has completed.')


val_im_list = glob.glob(root + '/leftImg8bit_trainvaltest/leftImg8bit/val/*')
counter = 0
for city in val_im_list:
    im_list = glob.glob(city + '/*.png')
    im_list.sort()
    for i in im_list:
        im = Image.open(i)
        im = im.resize((512, 256))
        im.save(im_val_root + '/{}.png'.format(counter))
        counter += 1
print('Validation RGB images processing has completed.')


# Disparity
train_im_list = glob.glob(root + '/disparity_trainvaltest/disparity/train/*')
counter = 0
for city in train_im_list:
    im_list = glob.glob(city + '/*.png')
    im_list.sort()
    for i in im_list:
        im = Image.open(i)
        im = im.resize((512, 256), resample=Image.NEAREST)
        im.save(depth_train_root + '/{}.png'.format(counter))
        counter += 1
print('Training depth images processing has completed.')


val_im_list = glob.glob(root + '/disparity_trainvaltest/disparity/val/*')
counter = 0
for city in val_im_list:
    im_list = glob.glob(city + '/*.png')
    im_list.sort()
    for i in im_list:
        im = Image.open(i)
        im = im.resize((512, 256), resample=Image.NEAREST)
        im.save(depth_val_root + '/{}.png'.format(counter))
        counter += 1
print('Validation depth images processing has completed.')


# Segmentation
counter = 0
train_label_list = glob.glob(root + '/gtFine_trainvaltest/gtFine/train/*')
for city in train_label_list:
    label_list = glob.glob(city + '/*_labelIds.png')
    label_list.sort()
    for l in label_list:
        im = Image.open(l)
        im = im.resize((512, 256), resample=Image.NEAREST)
        im.save(label_train_root + '/{}.png'.format(counter))
        counter += 1
print('Training Label images processing has completed.')


counter = 0
val_label_list = glob.glob(root + '/gtFine_trainvaltest/gtFine/val/*')
for city in val_label_list:
    label_list = glob.glob(city + '/*_labelIds.png')
    label_list.sort()
    for l in label_list:
        im = Image.open(l)
        im = im.resize((512, 256), resample=Image.NEAREST)
        im.save(label_val_root + '/{}.png'.format(counter))
        counter += 1
print('Validation Label images processing has completed.')


# Part Segmentation
counter = 0
train_label_list = glob.glob(root + '/gtFinePanopticParts_trainval/gtFinePanopticParts/train/*')
for city in train_label_list:
    label_list = glob.glob(city + '/*.tif')
    label_list.sort()
    for l in label_list:
        im = Image.open(l)
        im = im.resize((512, 256), resample=Image.NEAREST)
        im.save(part_train_root + '{}.tif'.format(counter))
        counter += 1
print('Training Label images processing has completed.')

counter = 0
val_label_list = glob.glob(root + '/gtFinePanopticParts_trainval/gtFinePanopticParts/val/*')
for city in val_label_list:
    label_list = glob.glob(city + '/*.tif')
    label_list.sort()
    for l in label_list:
        im = Image.open(l)
        im = im.resize((512, 256), resample=Image.NEAREST)
        im.save(part_val_root + '/{}.tif'.format(counter))
        counter += 1
print('Validation Label images processing has completed.')

