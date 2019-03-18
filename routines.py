# -*- coding: utf-8 -*-

import os
import functools
import numpy as np
import tensorflow as tf
import pandas as pd
rng = np.random.seed(100)
from sklearn.model_selection import train_test_split
from sklearn import metrics

from skimage import io, transform, exposure, filters, color
 

def misc_seg_metrics(pred, ann):    
    """ Compute the spec, sens, acc  between two ndarrays.
    """
    #if len(np.unique(pred)) == 2:
    acc = [((pred==j)*(ann==j)).sum(dtype='float')/(ann==j).sum() for j in np.unique(ann)] 
    acc.append((pred==ann).sum()/float(ann.size))  
    return acc 
    
def misc_jaccard_index(pred, ann):
    intersection = pred*ann
    union = np.maximum(pred, ann)
    return np.sum(intersection.astype(float))/np.sum(union)

def misc_metrics(gt, pr, ma=None, thrs= 0.5):
    assert(gt.size == pr.size)
    if ma is None:
        ma = np.ones(gt.shape, dtype=bool)
    ma = np.ravel(ma).astype(bool)
    gt = np.ravel(gt)
    pr = np.ravel(pr)  
    gt = gt[ma] 
    pr = pr[ma] 
    auc = metrics.roc_auc_score(gt, pr)  
    p = (pr>=thrs).astype(int)
    mets = misc_seg_metrics(p, gt)
    f1 = metrics.f1_score(gt, p)
    jac = misc_jaccard_index(p, gt) 
    return [auc, mets[0], mets[1], mets[2], f1, jac]
 
def read_df(fpath, data_dir):
    df_train = pd.read_csv(fpath)
    x_paths = df_train['im_name'].map(lambda s: os.path.join(data_dir,s))
    y_paths = df_train['gt_name'].map(lambda s: os.path.join(data_dir,s))
    return x_paths, y_paths

### Training-Validation splits
def trainval_splits(ftrain, data_dir, validation_split=0.2, random_state=rng):
    x_train_paths, y_train_paths = read_df(ftrain, data_dir) 
    return train_test_split(x_train_paths, y_train_paths, 
                            test_size=validation_split,
                            random_state=random_state)



def _process_pathnames(fname, lname, resize=None):  
    img = io.imread(fname)
    gt = io.imread(lname)
    if gt.ndim < 3:
        gt  = np.expand_dims(gt, -1)
    gt = gt[...,:1]
    gt = (gt > 0).astype(int) # binarize the ground-truth
    if resize is not None:
        img = transform.resize(img, resize)
        gt = transform.resize(gt, resize)
        gt = gt >= filters.threshold_otsu(gt)
    return img, gt 

### Data augmentation routines
def shift_img(img, gt, width_shift_range, height_shift_range, rotate_range): 
    if width_shift_range  or height_shift_range:
        if width_shift_range:
            width_shift_range = np.random.uniform(-width_shift_range * img.shape[1],
                                                   width_shift_range * img.shape[1])
        if height_shift_range:
            height_shift_range = np.random.uniform(-height_shift_range * img.shape[0],
                                                   height_shift_range * img.shape[0]) 
        tr = transform.AffineTransform(translation=(width_shift_range, height_shift_range )) 
        img = transform.warp(img, tr, preserve_range=True)
        gt   = transform.warp(gt, tr, preserve_range=True)
        
    if rotate_range :
        if isinstance(rotate_range, np.ScalarType):
            degre = np.random.uniform(-rotate_range,rotate_range)
        else:
            degre = np.random.uniform(rotate_range[0], rotate_range[1])
        img = transform.rotate(img, degre, preserve_range=True)
        gt  = transform.rotate(gt, degre, preserve_range=True)
        
    return img, gt

def flip_img(img, gt, horizontal_flip, vertical_flip):
    if horizontal_flip:
        flip_prob = np.random.uniform(0.0, 1.0)
        img, gt = (img, gt) if flip_prob >= 0.5 else (np.flip(img, 1), np.flip(gt, 1))
    if vertical_flip:
        flip_prob = np.random.uniform(0.0, 1.0)
        img, gt = (img, gt) if flip_prob >= 0.5 else (np.flip(img, 0), np.flip(gt, 0))
    return img, gt

def _process_img(img, gt, gamma=0,
                clahe=False, gray=False, xyz=False, hed=False,
                horizontal_flip=False, width_shift_range=0,
                height_shift_range=0, vertical_flip=0, rotate_range=0):
    img = exposure.rescale_intensity(img.astype(float), out_range=(0,1))
    if gray:
        img = color.rgb2gray(img)
    if xyz:
        img = color.rgb2xyz(img)
    if hed:
        img = color.rgb2hed(img)
    img = exposure.rescale_intensity(img, out_range=(0,1))
    if clahe:
        img = exposure.equalize_adapthist(img)
    if gamma:
        img = exposure.adjust_gamma(img, gamma)
        img = exposure.rescale_intensity(img, out_range=(0,1))
    if img.ndim == 2:
        img = np.expand_dims(img, -1) 

    img, gt = flip_img(img, gt, horizontal_flip, vertical_flip)    
    img, gt = shift_img(img, gt, width_shift_range, height_shift_range, rotate_range) 
         
    return img, gt
 

def image_generator(im_paths, gt_paths, 
              reader_fn=functools.partial(_process_pathnames),
              preproc_fn=functools.partial(_process_img),
              batch_size=1, 
              MAX_IM_QUEUE=20):  
    batch_x = []
    batch_y = []
    im_stack = dict() 
    while True:  
        for im_path, gt_path in zip(im_paths, gt_paths) :   
            hash_im = hash(im_path)
            if not im_stack.has_key(hash_im):
                img, gt = reader_fn(im_path, gt_path) 
                if len(im_stack.keys()) > MAX_IM_QUEUE:
                    im_stack.popitem()
                im_stack[hash_im] = (img, gt)
            else:
                img, gt = im_stack[hash_im] 

            pr_im, pr_gt  = preproc_fn(img, gt)   

            if len(batch_x) < batch_size:
                batch_x.append(pr_im)
                batch_y.append(pr_gt)
            else:
                ret = (np.array(batch_x), np.array(batch_y))
                batch_x, batch_y = [pr_im], [pr_gt]
                yield ret 


def get_image_generator(x_train_paths, y_train_paths, batch_size=1,
            width_shift_range=0, height_shift_range=0, 
            horizontal_flip=False,vertical_flip=False,
            rotate_range=0, resize=None,
            gamma=0, clahe=False, gray=False, xyz=False, hed=False,
            MAX_IM_QUEUE=100):
 
    prepro_cfg = dict(gamma=gamma, clahe=clahe,
                    gray=gray, xyz=xyz, hed=hed, horizontal_flip=horizontal_flip,
                    vertical_flip=vertical_flip, width_shift_range=width_shift_range,
                    height_shift_range=height_shift_range)
    prepro_fn = functools.partial(_process_img, **prepro_cfg)  

    reader_cfg = dict(resize=resize)
    reader_fn = functools.partial(_process_pathnames, **reader_cfg)

    return image_generator(x_train_paths, y_train_paths, reader_fn=reader_fn,
                    preproc_fn=prepro_fn, batch_size=batch_size, MAX_IM_QUEUE=MAX_IM_QUEUE)


class Patch_Sequence(tf.keras.utils.Sequence):
    def __init__(self, fixed_patch_ids, p_shape=(32,32,3),
                reader_fn=functools.partial(_process_pathnames),
                preproc_fn=functools.partial(_process_img),
                batch_size=32,
                MAX_IM_QUEUE=20, unsup=False):
        self.ids = fixed_patch_ids #
        self.p_shape = p_shape
        self.batch_size = batch_size
        self.reader_fn = reader_fn
        self.preproc_fn = preproc_fn
        self.MAX_IM_QUEUE = MAX_IM_QUEUE
        self.im_stack = {}
        self.unsup = unsup

    def __len__(self):
        return int(np.ceil(len(self.ids) / float(self.batch_size)))

    def __getitem__(self, idx):
        cur_id = self.ids[idx * self.batch_size:(idx + 1) * self.batch_size] 

        batch_x = []
        batch_y = []
        for pos in cur_id: 
            pid, pim, pgt = pos[2:], pos[0], pos[1] 
            x_p, y_p = pid.astype(int) 
            hash_im = hash(pim)
            if not self.im_stack.has_key(hash_im):
                img, gt = self.reader_fn(pim, pgt) 
                img, gt = self.preproc_fn(img, gt) 
                img = np.pad(img, ((self.p_shape[0]//2,), (self.p_shape[1]//2,), (0,)), mode='reflect')
                gt = np.pad(gt, ((self.p_shape[0]//2,), (self.p_shape[1]//2,), (0,)), mode='reflect')  
                if len(self.im_stack.keys()) > self.MAX_IM_QUEUE:
                    self.im_stack.popitem()
                self.im_stack[hash_im] = (img, gt)
            else:
                img, gt = self.im_stack[hash_im] 
            patch = img[x_p:x_p+self.p_shape[0], y_p:y_p+self.p_shape[1]] 
            label = gt[x_p:x_p+self.p_shape[0], y_p:y_p+self.p_shape[1]] 
            batch_x.append(patch)
            batch_y.append(label)
        if self.unsup:
            return np.array(batch_x), [np.array(batch_y), np.array(batch_x)]
        return np.array(batch_x), np.array(batch_y)
    
    def on_epoch_end(self, epoch=None, logs=None):
        self.im_stack = {} 

def get_patch_generator(dataset_ids, p_shape, batch_size=1, gamma=0., 
            clahe=False,  gray=False, xyz=False, hed=False,
            width_shift_range=0, height_shift_range=0, 
            horizontal_flip=False,vertical_flip=False,
            rotate_range=0, resize=None,
            MIN_PATCH_STD=None, MAX_IM_QUEUE=100,
            return_keras_queuer=False):
 
    prepro_cfg = dict(gamma=gamma, horizontal_flip=horizontal_flip,
                    vertical_flip=vertical_flip, width_shift_range=width_shift_range,
                    height_shift_range=height_shift_range, clahe=clahe, gray=gray, xyz=xyz, hed=hed)
    prepro_fn = functools.partial(_process_img, **prepro_cfg) 
    reader_cfg = dict(resize=resize)
    reader_fn = functools.partial(_process_pathnames, **reader_cfg)

    return Patch_Sequence(dataset_ids, p_shape=p_shape,
                        reader_fn=reader_fn, preproc_fn=prepro_fn,
                        batch_size=batch_size, MAX_IM_QUEUE=MAX_IM_QUEUE)
    
