import os

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
import cv2
from scipy.stats import circmean


def _hsv_dist(hsv0, hsv1):
    dh = min(abs(hsv0[0] - hsv1[0]), 1 - abs(hsv0[0] - hsv1[0]))
    ds = hsv0[1] - hsv1[1]
    dv = hsv0[2] - hsv1[2]
    return np.sqrt(2*dh**2 + 0.5*ds**2 + 0.1*dv**2)

def _hsv_mean(hsv):
    # [N, 3] cv2.HSV space
    hsv_mean = hsv.mean(0) / 255.
    hue = np.deg2rad(hsv[...,0] * 2)
    hue = np.rad2deg(circmean(hue)) / 360.
    hsv_mean[0] = hue
    return hsv_mean

def show_anns(image, anns, name, sort=True, labels=None, colormap=None):
    if len(anns) == 0:
        return
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True) if sort else anns
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for i, ann in enumerate(sorted_anns):
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        if labels is not None:
            l = labels[i]
        else:
            l = i + 1
        if colormap is not None:
            color_mask = np.asarray(matplotlib.colormaps[colormap].colors)[l]
        else:
            color_mask = np.random.random((1, 3)).tolist()[0]

        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.85)))
    plt.axis('off')
    plt.savefig(name)
    if not os.path.exists(name[:-4]):
        os.mkdir(name[:-4])
    for j, ann in enumerate(sorted_anns):
        plt.figure(figsize=(20, 20))
        plt.imshow(image)
        ax = plt.gca()
        ax.set_autoscale_on(False)
        m = ~ann['segmentation']
        ''''''
        img = np.ones((m.shape[0], m.shape[1], 3))
        for i in range(3):
            img[:, :, i] = 1
        ax.imshow(np.dstack((img, m * 0.)))
        plt.axis('off')
        plt.savefig(os.path.join(name[:-4], "{}.png".format(j)))

        img = np.zeros((m.shape[0], m.shape[1], 3))
        img[~m] = image[~m]
        cv2.imwrite(os.path.join(name[:-4], "{}.png".format(j)),
                    cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR))

def get_sam_everything_mask(image, image_mask, SamMaskGenerator, SamPredictor=None, device='cuda', min_area=500, 
                            min_color_similarity=0.25, feat_guidance=None, name='tmp_save/everything_sam', 
                            derender_dir='', colormap='Set3',
                            material_masks=None):
    ## input: image [h,w,3]
    ## input: image_mask [h, w] bool

    img_numpy = (image * 255).astype(np.uint8)

    if feat_guidance == 'hsv': # can change to any image features
        guide_numpy = cv2.cvtColor(img_numpy, cv2.COLOR_RGB2HSV).astype(np.float32)
        mean_func = _hsv_mean
        dist_func = _hsv_dist
    else: 
        guide_numpy = img_numpy.astype(np.float32)
        mean_func = lambda x : x.mean(0)/255.
        dist_func = 'euclidean'

    masks = SamMaskGenerator.generate(img_numpy)
    show_anns(img_numpy, masks, name + '_sam.png')

    if material_masks is None:
        mean_colors = []
        valid_masks = []
        full_segmentation = np.zeros_like(image_mask)
        for i, mask in enumerate(sorted(masks, key=(lambda x: x['area']), reverse=False)):
            seg = mask['segmentation'] & image_mask
            seg = seg ^ (full_segmentation & seg)
            area = seg.sum()
            if area < min_area:
                continue
            valid_masks.append({'area': area, 'segmentation': seg})
            full_segmentation = full_segmentation | seg
            mean_colors.append(mean_func(guide_numpy[seg]))

        mean_colors = np.stack(mean_colors)
        distances = pairwise_distances(mean_colors, metric=dist_func)
        show_anns(img_numpy, valid_masks, name + '_sam_valid.png', sort=False)

        merged_masks = []
        left_labels = set(range(len(mean_colors)))
        for i, dst in enumerate(distances):
            if i not in left_labels:
                continue
            to_merge_idx = set(np.where(dst < min_color_similarity)[0].tolist())
            to_merge_idx = to_merge_idx & left_labels
            left_labels = left_labels - to_merge_idx
            merged_seg = valid_masks[i]['segmentation']
            for j in to_merge_idx:
                merged_seg = merged_seg | valid_masks[j]['segmentation']
            merged_masks.append({'area': merged_seg.sum(), 'segmentation': merged_seg})
        merged_labels = list(range(1, len(merged_masks)+1))
        show_anns(img_numpy, merged_masks, name + '_sam_merged.png', sort=False, colormap=colormap)
    else:
        material_masks = np.load(material_masks).astype(np.uint8) * 255 # [num_class, h, w] bool
        material_masks = cv2.resize(material_masks.transpose(1,2,0), (image.shape[:2])).transpose(2,0,1)
        num_class = material_masks.shape[0]
        merged_labels = list(range(1, num_class+1))
        merged_masks = [{'segmentation': mat_mask > 128} for mat_mask in material_masks]
        show_anns(img_numpy, merged_masks, name + '_sam_merged.png', sort=False, colormap=colormap)

    merged_colors = [mean_func(guide_numpy[m['segmentation']]) for m in merged_masks]

    # visualize the color maps
    plt.figure(figsize=(20, 20))
    color_map = np.zeros((len(merged_colors) * 20, 20, 3))
    for i, c in enumerate(merged_colors):
        color_map[i*20:(i+1)*20,:20] = c
    plt.imshow((color_map * 255).astype(np.uint8))
    plt.savefig(name + '_colormap.png')

    print('==================== for derender ====================')

    if os.path.exists(derender_dir):
        merged_spec_terms = get_spatial_specular(img_numpy, derender_dir, merged_masks,  name + '_sam_specular.png', topn=0.1, bottomn=0.5)
    else:
        merged_spec_terms = None

    image_label = np.zeros(img_numpy.shape[:2])
    for label, mask in zip(merged_labels, merged_masks):
        image_label[mask['segmentation']] = label
    merged_labels, image_label = torch.from_numpy(np.asarray(merged_labels)).to(device), torch.from_numpy(image_label).to(device)

    return merged_labels, merged_masks, merged_colors, image_label, merged_spec_terms

def assign_sam_everything_mask(image, image_mask, SamMaskGenerator, device, dst_features, min_area=200, 
                               min_color_similarity=0.2, feat_guidance=None, name='tmp_save/everything_sam.png', 
                               colormap='Set3'):
    ## input: image [h,w,3]

    img_numpy = (image * 255).astype(np.uint8)

    if feat_guidance == 'hsv': # can change to any image features
        guide_numpy = cv2.cvtColor(img_numpy, cv2.COLOR_RGB2HSV).astype(np.float32)
        mean_func = _hsv_mean
        dist_func = _hsv_dist
    else: 
        guide_numpy = img_numpy.astype(np.float32)
        mean_func = lambda x : x.mean(0)/255.
        dist_func = 'euclidean'

    masks = SamMaskGenerator.generate(img_numpy)

    show_anns(img_numpy, masks, name + '_sam.png')

    mean_colors = []
    valid_masks = []
    full_segmentation = np.zeros_like(image_mask)
    for i, mask in enumerate(sorted(masks, key=(lambda x: x['area']), reverse=False)[:-1]): # the largest one is background
        seg = mask['segmentation'] & image_mask
        seg = seg ^ (full_segmentation & seg)
        area = seg.sum()
        if area < min_area:
            continue
        valid_masks.append({'area': area, 'segmentation': seg})
        full_segmentation = full_segmentation | seg
        mean_colors.append(mean_func(guide_numpy[seg]))

    mean_colors = np.stack(mean_colors)
    distances = pairwise_distances(mean_colors, dst_features, metric=dist_func)
    assigned_labels = distances.argmin(-1) + 1 # the valid label starts from 1
    distances_min = distances.min(-1)
    assigned_labels[distances_min > min_color_similarity] = 0 # threshold

    show_anns(img_numpy, valid_masks, name + '_valid_sam.png', sort=False, labels=assigned_labels, colormap=colormap)

    image_label = np.zeros(img_numpy.shape[:2])
    for label, mask in zip(assigned_labels, valid_masks):
        image_label[mask['segmentation']] = label
    image_label = torch.from_numpy(image_label).to(device) # other parts are default to be zero
    return image_label # [1,1,w,w]

def get_spatial_specular(image, derender_dir, anns, name, topn=0.1, bottomn=0.1):

    spec_img = cv2.imread(os.path.join(derender_dir, 'spec.jpg'))
    spec_img = cv2.resize(spec_img, image.shape[:2])
# try:
    final_spec_img = np.ones(spec_img.shape)
    merged_spec_terms = [0.,]
    for j, ann in enumerate(anns):
        m = ann['segmentation']
        part_spec = spec_img[m].mean(-1) / 255.
        N_top = int(topn * len(part_spec))
        N_bottom = int(bottomn * len(part_spec))
        spec_top = np.mean(sorted(part_spec)[-N_top:])
        # spec_mean = part_spec.mean()
        # part_spec = spec_top - spec_mean
        spec_bottom = np.mean(sorted(part_spec)[:N_bottom])
        part_spec = spec_top - spec_bottom
        final_spec_img[m] = part_spec
        merged_spec_terms.append(part_spec)
    cv2.imwrite(name, cv2.cvtColor(np.hstack([image, spec_img, (final_spec_img * 255).astype(np.uint8)]), cv2.COLOR_RGB2BGR))
# except:
#     print('failed')
    return np.asarray(merged_spec_terms)
