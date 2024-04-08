

import gradio as gr
import numpy as np
import datetime
import pathlib
import shlex
import subprocess
import time
import cv2
import trimesh
import pyrallis
from matplotlib import pyplot as plt
import torch.nn.functional as F
from PIL import Image
import torch
import gradio as gr
import torch.nn as nn
from typing import Any, Dict, List
from PIL import Image
from editing.src import utils
from editing.src.configs.train_config import GuideConfig, LogConfig, TrainConfig
from editing.src.training.trainer import Tex3D
import torchvision.transforms as transforms


DESCRIPTION = '''# HyperDreamer For Editing
NOTES: 
1) Upload the Mesh and Texture first, then confirm the front offset and click the button to init model
2) The Model default to render the front view. 
3) If you want to change the view, you can choose the (Phi, Theta, Radius) to render again to achieve ideal view after click the 'Reset' button.
4) The default label is positive(1). If you want to switch to another, please click the corresponding button first.
5) After SAM on single view, you need to confirm the (Edit Text, seed, scale and other args...) and click the Button(T) to Enter the following Edit Stage.
6) After All Processes are over, you can download the relative files in the lower left items corner.
7) Due to the long time cost in model initial, you can click 'Reset' to edit again in the current 3DMesh.
'''

class Model:
    def __init__(self):
        self.max_num_faces = 100000
        self.trainer = None
        self.config = None
        self.device = None
    def load_config(self, shape_path, train_grid_size) -> TrainConfig:
    
        log = LogConfig(exp_name=self.gen_exp_name())
        guide = GuideConfig()
        self.config = TrainConfig(log=log, guide=guide)
        self.config.guide.shape_path = shape_path
        self.config.render.train_grid_size = train_grid_size
     
    
    def init_model(self, shape_path, texture_img, train_grid_size):
        
        if not shape_path.endswith('.obj'):
            raise gr.Error('The input file is not .obj file.')

        yield None, None, 'The Model is Loading ......'
        

        self.load_config(shape_path, train_grid_size)
        self.device = self.config.optim.device
        self.trainer = Tex3D(self.config)
        texture_resolution = self.trainer.cfg.guide.texture_resolution
        if train_grid_size==1200:
            texture_resolution=1024
        else:
            texture_resolution=2048        
        texture =  transforms.Resize((texture_resolution, texture_resolution))(torch.Tensor(np.array(texture_img)).permute(2, 0, 1).unsqueeze(0) / 255.0).to(self.device)

        tex_image = nn.Parameter(texture)

        self.trainer.mesh_model.texture_img = tex_image
        self.trainer.mesh_model.original_texture_img = self.trainer.mesh_model.texture_img.clone().detach()
        self.trainer.mesh_model.meta_texture_img = nn.Parameter(torch.zeros_like(self.trainer.mesh_model.texture_img))

        print('Now render the first view Render_Image')
        
        theta = 60.0
        phi = 0.0
        radius = 1.5
        
        print('theta: ',theta,' phi: ',phi)
        
        theta = np.deg2rad(theta)
        phi = np.deg2rad(phi)
        theta = torch.FloatTensor([theta]).to(self.device)
        phi = torch.FloatTensor([phi]).to(self.device)
        

        phi = float(phi + 2 * np.pi if phi < 0 else phi)
        
        
        original_outputs = self.trainer.mesh_model.render(theta=theta, phi=phi, radius=radius, use_texture_original=True)
        sam_img = original_outputs['image']  #[1,3,w,w] tensor
      
        sam_img = utils.tensor_toImage(sam_img)  #[w,w,3] or [w,w,1]
        Original_img = sam_img
        yield sam_img , Original_img, 'The Model Loaded Successfully, you can SAM on a single view now'
        
    def gen_exp_name(self) -> str:
        now = datetime.datetime.now()
        return now.strftime('%Y-%m-%d-%H-%M-%S')

    def check_num_faces(self, path: str) -> bool:
        with open(path) as f:
            lines = [line for line in f.readlines() if line.startswith('f')]
        return len(lines) <= self.max_num_faces

    def zip_results(self, exp_dir: pathlib.Path) -> str:
        mesh_dir = exp_dir / 'mesh'
        out_path = f'{exp_dir.name}.zip'
        subprocess.run(shlex.split(f'zip -r {out_path} {mesh_dir}'))
        return out_path
        
        
    def render_single_view(self, theta, phi, radius, X, Y):  
        print('Now render the first view Render_Image')
        print('theta: ',theta,' phi: ',phi)
        
        theta = np.deg2rad(theta)
        phi = np.deg2rad(phi)
        theta = torch.FloatTensor([theta]).to(self.device)
        phi = torch.FloatTensor([phi]).to(self.device)

        phi = float(phi + 2 * np.pi if phi < 0 else phi)
        
        original_outputs = self.trainer.mesh_model.render(theta=theta, phi=phi, radius=radius, use_texture_original=True)
        rendered_img = original_outputs['image']  #[1,3,w,w] tensor
      
        rendered_img = utils.tensor_toImage(rendered_img)  #[w,w,3] or [w,w,1]
        
        X = None
        Y = None
        label_value = 1
        input_points = []
        input_labels = []
        Original_img = rendered_img
        print('reset label_value',label_value)
        print('reset input_points',input_points)
        print('reset input_labels',input_labels)
        
        return rendered_img, X, Y, label_value, input_points, input_labels, Original_img
    

    
    def run_text2tex(self, theta, phi, radius, input_points, input_labels, text, edit_model, 
            seed, guidance_scale, refine_threshold, strength, ddim_steps,
            generate_region_dialte_kernelsize, junction_region_dialte_kernelsize):
        
        start_time = time.time() 
        self.config.render.base_theta = theta
        self.config.render.radius = radius
        phi_initial = phi
        
        self.trainer.edit_step = 0
        self.trainer.mesh_model.train()
        
        print('---------------------Edit Model----------------: ', edit_model)
        self.config.optim.seed = seed
        self.config.guide.edit_text = text
        self.config.guide.guidance_scale = guidance_scale   
        self.config.guide.edit_model = edit_model   
        self.config.guide.refine_threshold = refine_threshold   
        self.config.guide.strength = strength   
        self.config.guide.ddim_steps = ddim_steps   
        self.config.guide.generate_region_dialte_kernelsize = generate_region_dialte_kernelsize   
        self.config.guide.junction_region_dialte_kernelsize = junction_region_dialte_kernelsize  

        pyrallis.dump(self.trainer.cfg, (self.trainer.exp_path / 'config.yaml').open('w'))
        
        self.trainer.dataloaders = self.trainer.init_dataloaders()
        
        
        points = np.array(input_points)
        labels = np.array(input_labels)
        
        print('====================================')
        print('points: ', points)
        print('labels: ', labels)
        #------------------- SAM Step ---------------#
        total_steps = len(self.trainer.dataloaders['segment'])
        for step, data in enumerate(self.trainer.dataloaders['segment']):
            
            self.trainer.edit_step += 1
            if self.trainer.edit_step == 1:
                phi_0 = data['phi']
            phi_offset = phi_initial - phi_0
            
            if self.trainer.edit_step == 1:
                single_view_mask, original_rgb_render = self.trainer.generate_3D_SAM(data, points=points, labels=labels, phi_offset=phi_offset) #[1,3,w,w]
            else:
                single_view_mask, original_rgb_render = self.trainer.generate_3D_SAM(data, phi_offset=phi_offset)

            print(f"------------Segment[{self.trainer.edit_step}] Done-------------")
            single_view_mask = torch.sum(single_view_mask.squeeze(0),dim=0)  #[w,w]
            single_view_mask = (single_view_mask > 0).float().unsqueeze(0).unsqueeze(0).to(self.device) #[1,1,w,w]
            
            original_rgb_render = original_rgb_render.permute(0,2,3,1).squeeze(0).cpu().detach().numpy()  #[w,w,3] 
            single_view_mask = single_view_mask.permute(0,2,3,1).squeeze().cpu().detach().numpy()  #[w,w]
            
            h, w = single_view_mask.shape[0], single_view_mask.shape[1]

            color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
            color =  color.reshape(1, 1, -1)
            single_view_mask = single_view_mask.astype(bool)
            
            single_view_result = np.ones((h,w,4))
            single_view_result[:, :, :3] = original_rgb_render

            single_view_result[single_view_mask, :] = single_view_result[single_view_mask, :] * color
            
            single_view_result = (single_view_result.copy() * 255).astype(np.uint8)
            name = "SAM_view_result_gradio"
            Image.fromarray(single_view_result).save(self.trainer.edit_path / f'{self.trainer.edit_step:04d}_{name}.png')
            
            yield single_view_result, None, None, None, f'Now SAM for the 3D Mesh: {self.trainer.edit_step}/{total_steps}'
            
        endsam_time = time.time() 
        elapsedsam_time_seconds = endsam_time - start_time
        minutes_sam = int(elapsedsam_time_seconds // 60)
        seconds_sam = int(elapsedsam_time_seconds % 60)
        #mask_texture_img = self.trainer.mesh_model.mask_texture_img  #[1,3,w,w]
        
        self.trainer.evaluate(self.trainer.dataloaders['val_large'], self.trainer.final_edit_renders_path, save_as_video=True,use_texture_mask=True)
        mask_video_path = self.trainer.final_edit_renders_path / 'result_mask.mp4'  
        
        yield single_view_result, mask_video_path.as_posix(), None, None, f'SAM For 3D Mesh Done! Time spent: {minutes_sam}min {seconds_sam}s'

        #------------------- Edit Step ---------------#
        self.trainer.edit_step = 0
        self.trainer.mesh_model.train()
        self.trainer.mesh_model.preparing_editing_texture()
        print('-------------- NOW Editing the Initial 3D Object --------------')

        for step, data in enumerate(self.trainer.dataloaders['train']):
            self.trainer.edit_step += 1
            single_view_edit = self.trainer.edit_viewpoint(data)
            single_view_edit = utils.tensor_toImage(single_view_edit)  #[w,w,3] or [w,w,1]
            
            yield single_view_edit, mask_video_path.as_posix(), None, None, f'Now Edit for the 3D Mesh: {self.trainer.edit_step}/{total_steps}'
        
        self.trainer.edit_step = 0
        self.trainer.evaluate(self.trainer.dataloaders['train'], self.trainer.edit_eval_path)
        
        self.trainer.full_eval(self.trainer.final_edit_renders_path)
        mask_edit_video_path = self.trainer.final_edit_renders_path / 'result_rgb.mp4'  
        
        zip_path = self.zip_results(self.trainer.final_edit_renders_path)
        
        save_dir = self.trainer.final_edit_renders_path / 'mesh'
        save_dir.mkdir(exist_ok=True, parents=True)
        self.trainer.mesh_model.export_mesh(save_dir)
        # model_path = save_dir / 'mesh.obj'
        # mesh = trimesh.load(model_path)
        # mesh_path = save_dir / 'mesh.glb'
        # mesh.export(mesh_path, file_type='glb')
        end_time = time.time()
        elapsed_time_seconds = end_time - start_time
        minutes = int(elapsed_time_seconds // 60)
        seconds = int(elapsed_time_seconds % 60)
        yield single_view_edit, mask_video_path.as_posix(), mask_edit_video_path.as_posix(),  zip_path, f'Edit for 3dMesh all done!! Total spent: {minutes}min {seconds}s'
            

def label_change_to_1(label_value):
    label_value=1
    print('label_value: ',label_value)
    return label_value

def label_change_to_0(label_value):
    label_value=0
    print('label_value: ',label_value)
    return label_value

def reset_label(X, Y, label_value, input_points, 
                input_labels, Original_img):
    X = None
    Y = None 
    label_value = 1
    input_points = []
    input_labels = []
    sam_img = Original_img
    print('reset label_value',label_value)
    print('reset input_points',input_points)
    print('reset input_labels',input_labels)
    return sam_img, X, Y, label_value, input_points, input_labels


def reset_all(*args):
    
    results = [None for _ in args]
    results[0] = 'Reset the all current editing setting, but you can continue to upload files or choose params to Edit new'
    results[1] = 1
    return results


model = Model()


with gr.Blocks(css='style.css') as demo:
    gr.Markdown(DESCRIPTION)
    input_points = gr.State([])
    input_labels = gr.State([])
    Original_img = gr.State(None)
    
    label_1 = gr.State(value = 1)
    label_0 = gr.State(value = 0)
    label_value = gr.State(value = 1)
    All_edit_counts = gr.State(value = 1)
    
    with gr.Row():                
        with gr.Column():
            gr.Markdown("## Model Initial")  
            with gr.Row():
                input_shape = gr.Model3D(label='Upload 3D Mesh')
                input_texture = gr.Image(source='upload', type="pil", label="Upload Texture Image")
                                
            train_grid_size = gr.Radio([1200, 2400], label="Train Grid Size", value = 2400)
         
            run_init_button = gr.Button('Run HyperDreamer For Editing', variant="primary")
            reset_all_button = gr.Button('Reset All')

            gr.Markdown("## Processing Results") 
            progress_text = gr.Text(label='Progress')
            with gr.Tabs():
                with gr.TabItem(label='Each View Image'):
                    viewpoint_img = gr.Image(show_label=False)
                with gr.TabItem(label='Result Mask Video'):
                    result_mask_video = gr.Video(show_label=False)
                with gr.TabItem(label='Result Edit Video'):
                    result_edit_video = gr.Video(show_label=False)

                with gr.TabItem(label='Output Mesh Files'):
                    output_file = gr.File(show_label=False)
                    

        with gr.Column():   
            with gr.Tabs():
                with gr.TabItem(label='Local Text-Based Texture Editing'):
                    with gr.Column():
                        gr.Markdown("## SAM For single view")          
                        #progress_text = gr.Text(label='Progress')
                        sam_img = gr.Image(label = 'Interactive SAM')
                        with gr.Column():
                            with gr.Row():
                                select_label_1 = gr.Button('Positive Label')
                                select_label_0 = gr.Button('Negative Label')
                            with gr.Row():
                                X = gr.Number(label="Point Coordinate X")
                                Y = gr.Number(label="Point Coordinate Y")
                    
                    
                    text = gr.Text(label='Editing Instruct Text')
                    with gr.Accordion("Advanced options", open=False):
                        
                        edit_model = gr.Radio(["Normal_Aware", "Depth_Aware"], label="Editting Model", value = "Normal_Aware")
                        seed = gr.Slider(label='Seed',
                                        minimum=0,
                                        maximum=100000,
                                        value=25,
                                        step=1)
                        guidance_scale = gr.Slider(label='Editing Guidance scale',
                                                minimum=0,
                                                maximum=50,
                                                value=7.5,
                                                step=0.1)
                        
                        refine_threshold = gr.Slider(label='Threshold for defining refine regions',
                                                minimum=0,
                                                maximum=1,
                                                value=0.2,
                                                step=0.02)
                        
                        strength = gr.Slider(label='Control Strength',
                                                minimum=0.0,
                                                maximum=2.0,
                                                value=1.0,
                                                step=0.01)
                        
                        ddim_steps = gr.Slider(label='Number of denoising steps',
                                minimum=0,
                                maximum=500,
                                value=20,
                                step=1)
                        
                        generate_region_dialte_kernelsize = gr.Slider(label='Dialte kernel size: Generate_regions',
                                minimum=3,
                                maximum=80,
                                value=5,
                                step=1)
                        
                        junction_region_dialte_kernelsize = gr.Slider(label='Dialte kernel size: Junction(J⊆R)_regions',
                                minimum=3,
                                maximum=80,
                                value=20,
                                step=1)
                            
                    Edit_start_text2tex = gr.Button('Enter Text2tex Editing', variant="primary")        
                    
              
                    
            with gr.Row():             
                phi = gr.Slider(label='Phi',
                                    minimum=0,
                                    maximum=360,
                                    value=0,
                                    step=1)
                    
                theta = gr.Slider(label='Theta',
                                    minimum=0,
                                    maximum=180,
                                    value=60,
                                    step=1)
                            
                radius = gr.Slider(label='Radius',
                                    minimum=1,
                                    maximum=1.8,
                                    value=1.5,
                                    step=0.1)       
            run_render_button = gr.Button('Render the Single View')  
            Reset = gr.Button('Reset') 
                                


        
    def sam_interactive(label_value, 
                        input_points,
                        input_labels, 
                        Original_img, 
                        evt: gr.SelectData):

        
        img_sam = np.array(Original_img, dtype=np.uint8)
        x, y = evt.index[0], evt.index[1]
        input_points.append([x, y])
        input_labels.append(label_value)
        print('input_points',input_points)
        print('input_labels',input_labels)
         
        sam_tensor = utils.Image_totensor(img_sam) #[1,3,w,w]
        img_h ,img_w = sam_tensor.shape[2], sam_tensor.shape[3]
        now_sam_mask = model.trainer.sam3d.get_sam_mask(sam_tensor,
                                                  model.trainer.sam3d.predictor,
                                                  np.array(input_points),
                                                  np.array(input_labels),
                                                  img_h,
                                                  img_w)   #[1,1,w,w]
        now_sam_mask = utils.tensor_toImage(now_sam_mask)  #[w,w,1]
        now_sam_mask = now_sam_mask.squeeze() #[w,w]
        h, w = now_sam_mask.shape[-2:]
        mask_img = now_sam_mask.reshape(h, w, 1) 
        
        coords = np.array(input_points)
        labels = np.array(input_labels)
        for i in range(len(coords)):
            print(labels[i])
            # Set the points(circle color) based on the label
            color = (255, 0, 0) if labels[i] == 0 else (0, 0, 255)
            # Draw the points
            Xp=coords[i, 0]
            Yp=coords[i, 1]
            img_sam = cv2.circle(img_sam, (Xp, Yp), 20, color, -1)
                    
        # Set the opacity for the mask_image and edited_image
        opacity_mask = 0.6
        opacity_edited = 1.0

        # Combine the edited_image and the mask_image using cv2.addWeighted()
        img_sam = cv2.addWeighted(
            img_sam,
            opacity_edited,
            (mask_img *
             np.array([30 / 255, 144 / 255, 255 / 255])).astype(np.uint8),
            opacity_mask,
            0,
        )
        
        return img_sam, x, y, input_points, input_labels, label_value
        


    sam_img.select(sam_interactive, [label_value, input_points,
                                     input_labels, Original_img], 
                                    [sam_img, X, Y,
                                    input_points,
                                    input_labels,
                                    label_value])

    select_label_1.click(fn=label_change_to_1,inputs=[label_value],outputs=[label_value])
    select_label_0.click(fn=label_change_to_0,inputs=[label_value],outputs=[label_value])
    
    Reset.click(fn=reset_label,inputs=[X, Y, 
                                       label_value, input_points, 
                                       input_labels, Original_img],
                               outputs=[sam_img, X, Y,
                                        label_value, input_points, 
                                        input_labels])
    
    
    reset_all_button.click(fn=reset_all,inputs=[progress_text, label_value, input_points, input_labels,
                                                input_shape, input_texture, sam_img,
                                                X, Y, viewpoint_img, result_mask_video, 
                                                result_edit_video, output_file],
                                        outputs=[progress_text, label_value, input_points, input_labels,
                                                input_shape, input_texture, sam_img,
                                                X, Y, viewpoint_img, result_mask_video, 
                                                result_edit_video, output_file])
    
    
    run_init_button.click(fn=model.init_model,
                            inputs=[input_shape, 
                                    input_texture,
                                    train_grid_size],
                            outputs=[sam_img,
                                     Original_img,
                                     progress_text])

    run_render_button.click(fn=model.render_single_view,
                            inputs=[theta,
                                    phi,
                                    radius,
                                    X,
                                    Y],
                            outputs=[sam_img,
                                     X,
                                     Y,
                                     label_value,
                                     input_points,
                                     input_labels,
                                     Original_img])
    
    
    Edit_start_text2tex.click(fn = model.run_text2tex,
                     inputs=[theta,
                             phi,
                             radius,
                             input_points,
                             input_labels,
                             text,
                             edit_model,
                             seed,
                             guidance_scale,
                             refine_threshold,
                             strength,
                             ddim_steps,
                             generate_region_dialte_kernelsize,
                             junction_region_dialte_kernelsize],
                     outputs=[viewpoint_img,
                              result_mask_video,
                              result_edit_video,
                              output_file,
                              progress_text
                             ])
    

    
demo.queue(max_size=5).launch(server_name='10.140.1.136',server_port=10088,share=True)
