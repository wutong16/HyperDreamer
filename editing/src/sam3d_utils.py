import torch
import torchvision
import numpy as np
from PIL import Image
from loguru import logger
from sklearn.metrics import pairwise_distances
from editing.src.configs.train_config import TrainConfig
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator


class SAM3D_Mesh:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.predictor = self.sam_load()

    def sam_load(self):
        sam_checkpoint = "models/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(self.device)
        predictor = SamPredictor(sam)
        # if use_everything:
        #     predictor = SamAutomaticMaskGenerator(
        #                                     model=sam,
        #                                     points_per_side=32,
        #                                     pred_iou_thresh=0.85,
        #                                     stability_score_thresh=0.85,
        #                                     crop_n_layers=1,
        #                                     crop_n_points_downscale_factor=2,
        #                                     min_mask_region_area=100)  # Requires open-cv to run post-processing

        logger.info('Successfully load SAM Predictor')
        return predictor
    
    def find_mask(self, image, color):
        """
        Find mask for a specific color in the given image.
        :param image: input image tensor of shape [1, 3, H, W]
        :param color: color array of shape [3]
        :return: tensor of mask for the color of shape [1, 1, H, W]
        """
                
        h,w = image.shape[2],image.shape[3]
        color = color.reshape(1,-1,1,1)
        color = torch.from_numpy(color).float().to(self.device)
        color = color.repeat(1,1,h,w)
        # Compute squared euclidean distance between each pixel and the color
        dist = torch.sum((image*255 - color*255)**2, dim=1)
        dist = dist / 255
        mask = (dist < 0.1).float()
        mask = mask.to(self.device)
        return mask

        
        
    def get_sam_mask(self, image, SamPredictor, input_point, input_label, H=None, W=None):
        ## input: image [1,3,w,w]

        if H is None or W is None :
            H = self.cfg.render.train_grid_size
            W = self.cfg.render.train_grid_size
        torchvision.utils.save_image(image,'image.jpg')

        img_numpy = image.permute(0,2,3,1).squeeze(0).cpu().detach().numpy()
        img_numpy = (img_numpy.copy() * 255).astype(np.uint8)
        
        SamPredictor.set_image(img_numpy)
        
        if len(input_label) > 0:

            masks, scores, logits = SamPredictor.predict(
                                                        point_coords=input_point,
                                                        point_labels=input_label,
                                                        multimask_output=True)

            mask_input = logits[np.argmax(scores), :, :] 

            #print('mask_input',mask_input[None, :, :].shape)
            mask, _, _ = SamPredictor.predict(
                                                point_coords=input_point,
                                                point_labels=input_label,
                                                mask_input=mask_input[None, :, :],
                                                multimask_output=False)
            
            mask = torch.from_numpy(mask.copy()).float().to(self.device)
            
            mask = mask.unsqueeze(0)
        else:
            mask = torch.zeros(1,1,H,W).to(self.device)
        return mask  #[1,1,w,w]


    def mask_to_SAMInput(self, mask, negmask):
        
        #mask [1200,1200]
        Width = mask.shape[0]                #default: 1200  
        input_point = []
        input_label = []
        patch_size_pos = 48    #48(1200)   144(3600)    96(2400)    25*25
        patch_size_neg = 240    #240(1200)  720(3600)    480(2400)   5 * 5
        print('patch_size_pos:',patch_size_pos)
        print('patch_size_neg:',patch_size_neg)
        
        pos_allnums = len(torch.nonzero(mask,    as_tuple=False))
        neg_allnums = len(torch.nonzero(negmask, as_tuple=False))
        thed = 50 * (int(Width / 1200)**2)
        if pos_allnums > thed:
            patch_counts_pos = int(Width / patch_size_pos)
            patch_counts_neg = int(Width / patch_size_neg)        
            #patch_nums_pos = patch_counts_pos * patch_counts_pos # Number of pixel points in the patch
            #patch_nums_neg = patch_counts_neg * patch_counts_neg # Number of pixel points in the patch
            
            for i in range(patch_counts_pos):
                for j in range(patch_counts_pos):
                    m = torch.zeros_like(mask)
                    m[(i*patch_size_pos):((i+1)*patch_size_pos), (j*patch_size_pos):((j+1)*patch_size_pos)] = 1
                    mask_patch = mask.detach().clone()
                    mask_patch = mask_patch * m
                    index_pos = torch.nonzero(mask_patch, as_tuple=False)
                    # filp x and y 
                    index_pos = torch.flip(index_pos, [1])
                    index_pos = index_pos.cpu().detach().numpy()
                    index_pos_num = len(index_pos)
                    #pos_percentage = index_pos_num / patch_nums_pos

                    if  index_pos_num > thed:  #50
                        index_k = np.random.randint(0, index_pos_num)
                        input_point.append(index_pos[index_k].tolist())
                        input_label.append(1)
            
            for i in range(patch_counts_neg):
                for j in range(patch_counts_neg):
                    
                    m = torch.zeros_like(mask)
                    negmask_patch = negmask.detach().clone()   
                    m[(i*patch_size_neg):((i+1)*patch_size_neg), (j*patch_size_neg):((j+1)*patch_size_neg)] = 1
                    negmask_patch = negmask_patch * m
                    index_neg = torch.nonzero(negmask_patch, as_tuple=False)

                    # filp x and y 
                    index_neg = torch.flip(index_neg, [1])
                    index_neg = index_neg.cpu().detach().numpy()
                    index_neg_num = len(index_neg)
                    #neg_percentage = index_neg_num / patch_nums_neg
                    if  index_neg_num > thed:   #50
                        index_k = np.random.randint(0, index_neg_num)
                        input_point.append(index_neg[index_k].tolist())
                        input_label.append(0)
        
            input_point = np.array(input_point)
            input_label = np.array(input_label)
        
        print('num_all: ',len(input_point))
        print('num_pos:', np.sum(input_label==1))
        print('num_neg:', np.sum(input_label==0))
        
        print('+------------------------------+')
        return input_point, input_label
        

    def SAMInput_to_mask(self, input_point, input_label, H=None, W=None):
        if H is None or W is None :
            H = self.cfg.render.train_grid_size
            W = self.cfg.render.train_grid_size
            
        mask = torch.zeros(1, 1, H, W)
        counts = len(input_label)
        for k in range(counts):
            i = int(input_point[k, 1])
            j = int(input_point[k, 0])
            value = int(input_label[k])
            mask[:, :, i, j] = value
        mask = mask.to(self.device)
        return mask #[1,1,w,w]
    
    def _merge_colors(self,color_list):
        color = np.zeros(3)
        area = 0
        for c, a in color_list:
            color += a/1000 * c
            area += a/1000
        return color / area
    
    def _merge_masks(self,src_mask, new_mask):
        src_mask['segmentation'] = src_mask['segmentation'] + new_mask['segmentation']
        src_mask['area'] = src_mask['area'] + new_mask['area']
        # src_mask['point_coords'] = src_mask['point_coords'].append(new_mask['point_coords'])
        # the other attributes are not changed.
        return src_mask
    
    def get_sam_everything_mask(self, image, object_mask, SamMaskGenerator, device, min_area=1000, min_color_similarity=0.25):
        ## input: image [1,3,w,w]

        #torchvision.utils.save_image(image, name + '.png')

        background_mask = torch.abs(1-object_mask).squeeze().detach().cpu().numpy()  #[w,w]
        object_mask = object_mask.squeeze().detach().cpu().numpy()  #[w,w]
                
        img_numpy = image.permute(0, 2, 3, 1).squeeze(0).cpu().detach().numpy()
        img_numpy = (img_numpy.copy() * 255).astype(np.uint8)

        masks = SamMaskGenerator.generate(img_numpy)

    # try:
        mean_colors = []
        valid_masks = []
        full_segmentation = np.zeros((img_numpy.shape[:2])) > 0
        for i, mask in enumerate(sorted(masks, key=(lambda x: x['area']), reverse=False)):
            mask['segmentation'][mask['segmentation']] = mask['segmentation'][mask['segmentation']] ^ full_segmentation[mask['segmentation']]
            mask['area'] = (mask['segmentation'] * object_mask).sum()
            if mask['area'] < min_area or img_numpy[mask['segmentation']].mean() > 0.98 * 255:
                continue
            valid_masks.append(mask)
            full_segmentation += mask['segmentation']
            mean_colors.append(img_numpy[mask['segmentation']].mean(0))

        mean_colors = np.stack(mean_colors) / 255.
        distances = pairwise_distances(mean_colors, metric='euclidean')

        merged_masks = []
        merged_labels = []
        merged_colors = []
        left_labels = np.arange(len(mean_colors)).tolist()
        i_label = 1
        full_segmentation = np.zeros((img_numpy.shape[:2])) > 0
        for i, dst in enumerate(distances):
            if i not in left_labels:
                continue # when the area has already been merged
            if full_segmentation[valid_masks[i]['segmentation']].mean() > 0.98: # when the area is duplicated
                #print('For the {}th mask, most area have been covered. Skip it.'.format(i))
                continue
            to_merge_idx = np.where(dst < min_color_similarity)[0]
            to_merge_idx = to_merge_idx[to_merge_idx>i]
            merged_mask = valid_masks[i]
            merged_color = [[mean_colors[i], valid_masks[i]['area']]]
            merged_idx = []
            if len(to_merge_idx) > 0:
                for j in to_merge_idx:
                    if j not in left_labels:
                        continue
                    merged_mask = self._merge_masks(merged_mask, valid_masks[j])
                    merged_idx.append(j)
                    left_labels.remove(j)
                    merged_color.append([mean_colors[j], valid_masks[j]['area']])


            merged_masks.append(merged_mask)
            merged_labels.append(i_label)
            merged_colors.append(merged_color)
            i_label += 1
            full_segmentation += merged_mask['segmentation']

        
        merged_colors = [self._merge_colors(c) for c in merged_colors]


        image_label = np.zeros(img_numpy.shape[:2])
        for label, mask in zip(merged_labels, merged_masks):
            image_label[mask['segmentation']] = label
        merged_labels, image_label = torch.from_numpy(np.asarray(merged_labels)).to(device), torch.from_numpy(image_label).to(device)
        merged_labels = merged_labels.tolist()
        #print(merged_labels)
        return merged_labels, merged_masks, merged_colors, image_label
    
    def save_sam_img(self, single_view_mask, original_rgb_render, path=None, now_step=None, name='SAM_view_result'):
        #single_view_mask [1,3,w,w]

        single_view_mask = torch.sum(single_view_mask.squeeze(0),dim=0)  #[w,w]
        single_view_mask = (single_view_mask > 0).float().unsqueeze(0).unsqueeze(0).to(self.device) #[1,1,w,w]
        
        original_rgb_render = original_rgb_render.permute(0,2,3,1).squeeze(0).cpu().detach().numpy()  #[w,w,3] 
        single_view_mask = single_view_mask.permute(0,2,3,1).squeeze().cpu().detach().numpy()  #[w,w]
        
        h, w = single_view_mask.shape[0], single_view_mask.shape[1]

        color = np.array([112 / 255.0, 173 / 255.0, 71 / 255.0, 0.82])
        #color = np.array([140 / 255, 206 / 255, 238 / 255, 0.82])
        color =  color.reshape(1, 1, -1)
        single_view_mask = single_view_mask.astype(bool)
        
        single_view_result = np.ones((h,w,4))
        single_view_result[:, :, :3] = original_rgb_render

        single_view_result[single_view_mask, :] = single_view_result[single_view_mask, :] * color
        
        single_view_result = (single_view_result.copy() * 255).astype(np.uint8) #np [w,w,4]
        Image.fromarray(single_view_result).save(path / f'{now_step:04d}_{name}.png')