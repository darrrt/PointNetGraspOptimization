import argparse
import json
import math
import os.path
import time
import sys
import shutil

import scipy.sparse
import scipy.spatial
import trimesh.sample

import torch
import plotly.graph_objects as go
from utils.visualize_plotly import plot_point_cloud, plot_point_cloud_cmap, plot_mesh_from_name
from utils.set_seed import set_global_seed
from torch.utils.tensorboard import SummaryWriter
import trimesh as tm
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import point2grasp
import scipy 

objects_config={
    'contactdb+alarm_clock':{
        "mesh_file":os.path.join('data/object', "contactdb+alarm_clock".split('+')[0], "contactdb+alarm_clock".split("+")[1],f'{"contactdb+alarm_clock".split("+")[1]}.stl'),
        "pre_defined_feature":{
            # "left_ear":[x,y,z,r,nx,ny,nz] 法向量
            "back_mid":[[0.02,0.02,0,0.03,0,-1,0]],
            "front_mid":[[0,-0.01,0,0.02,0,1,0]],
            
        }
    },
    "ycb+bleach_cleanser":{},
    "cracker_box":{},
    "ycb+hammer":{},
    "ycb+master_chef_can":{},
    "ycb+mustard_bottle":{},
    "ycb+phillips_screwdriver":{},
    "ycb+pitcher_base":{},
    "ycb+power_drill":{},
    
    "contactdb+airplane":{},
    "contactdb+binoculars":{},
    "contactdb+bowl":{},
    "contactdb+camera":{},
    "contactdb+cup":{},
    "contactdb+door_knob":{},
    "contactdb+eyeglasses":{},
    "contactdb+flashlight":{},
    "contactdb+hammer":{},
    "contactdb+hand":{},
    "contactdb+headphones":{},
    "contactdb+knife":{},
    "contactdb+light_bulb":{},
    "contactdb+mouse":{},
    "contactdb+mug":{},
    "contactdb+pan":{},
    "contactdb+ps_controller":{},
    "contactdb+scissors":{},
    "contactdb+stapler":{},
    "contactdb+toothbrush":{},
    "contactdb+toothpaste":{},
    "contactdb+utah_teapot":{},
    "contactdb+water_bottle":{},
    "contactdb+wine_glass":{},
    "contactdb+wristwatch":{},
}
demands_test={
    'contactdb+alarm_clock':{
        "enhance":
            ['back_mid','front_mid'],
            # [[0.005,0.01,0.04,0.01,-0.05,0.003,-0.05],[0.003,0,0.05,0.01,0,0,-1],'back_mid','front_mid'],
        "weaken":[],
        "grasp_suggestion_palm_ori":[90,180,0],
        "grasp_suggestion_finger_joints":[0,45,45,45]+[0,90,0,0]*4,
        "grasp_description":"Hold the clock with your thumb and four fingers. The thumb is on the front of the clock and the four fingers are on the side of the clock."
    },
   
}
class Points2PointCloud():
    def __init__(self,objects_config=objects_config,device='cuda') -> None:
        self.objects_config=objects_config
        self.device=device
        pass
    def calcContactArea(self,count=2048):
        # enhaceArea=self.enhance_area
        # weakenArea=self.weaken_area
        focus_area=np.concatenate([self.weaken_area,self.enhance_area],axis=0)
        # 正的覆盖负的
        mapping_signs=np.ones(self.enhance_area.shape[0]+self.weaken_area.shape[0])*0.5
        mapping_signs[:self.weaken_area.shape[0]]*=-0.5
        
        point_cloud=self.object_point_cloud
        # 应该是不接触的点都为0，希望接触的点为1
        # 给定一个空间中的坐标点，和希望按压的朝向，对这个左边点做球，在半径内的都是希望接触的地方，同时需要考虑法向量，法向量夹角大于180度，距离其中
        # 球内赋值为最近点/距离 这样范围都在（0,1），多个球重合时保留最大值
        # 不在球内的点为0 希望接触的为0.5+xxx 不希望接触的为0.5-xxx xxx幅值为0.5
            
        contact_map=np.zeros((count,1))
        
        euclid_distance=scipy.spatial.distance.cdist(focus_area[:,:3],point_cloud[:,:3])
        cosin_distance=np.matmul(focus_area[:,4:7],point_cloud[:,3:].T)
        # print('distance',euclid_distance.shape,euclid_distance)
        for idx in range(focus_area.shape[0]):
            indices=np.where(np.bitwise_and(euclid_distance[idx,:]<focus_area[idx,3] , cosin_distance[idx,:]<0))
            min_distance=np.min(euclid_distance[idx,indices])
            # print('min_distance',min_distance,contact_map[indices],(0.5+mapping_signs[idx]*min_distance/euclid_distance[idx,indices]).reshape(-1,1))
            contact_map[indices]=np.maximum(contact_map[indices],(0.5+mapping_signs[idx]*min_distance/euclid_distance[idx,indices]).reshape(-1,1))
        return contact_map


    def genPointCloud(self,objects_list:list,demands:dict,cmap_path:str,vis_dir:str,if_post_process=False):
        cmap = []
        for object_name in objects_list:
            
            object_mesh: tm.Trimesh
            object_mesh = tm.load(self.objects_config[object_name]["mesh_file"])
            
            cmap_sample = {'object_name': object_name,
                            'object_point_cloud': None,
                            'contact_map_value': None,
                            "enhance":demands[object_name]['enhance'],
                            "weaken":demands[object_name]['weaken'],
                            "grasp_suggestion_palm_ori":torch.tensor(demands[object_name]['grasp_suggestion_palm_ori'],dtype=torch.float),
                            "grasp_suggestion_finger_joints":torch.tensor(demands[object_name]['grasp_suggestion_finger_joints'],dtype=torch.float),
                            "grasp_description":demands[object_name]['grasp_description'],
                            "mesh_file":self.objects_config[object_name]["mesh_file"]
                            }
            object_point_cloud, faces_indices = trimesh.sample.sample_surface_even(mesh=object_mesh, count=2048)
            # 当前采样点所在面的法向量
            contact_points_normal = [object_mesh.face_normals[x] for x in faces_indices]
            
            print('object_point_cloud',object_point_cloud.shape)
            self.object_point_cloud = np.concatenate([object_point_cloud, contact_points_normal], axis=1)
            print('object_point_cloud_normal',self.object_point_cloud.shape)
            # z_latent_code = torch.randn(1, model.latent_size, device=self.device).float()
            # contact_map_value = model.inference(object_point_cloud[:, :3].unsqueeze(0), z_latent_code).squeeze(0)
            # # process the contact map value
            # contact_map_value = contact_map_value.detach().cpu().unsqueeze(1)
            enhance_area=[]
            for demand in demands[object_name]['enhance']:
                if isinstance(demand,str):
                    for li in self.objects_config[object_name]['pre_defined_feature'][demand]:
                        enhance_area.append(li)
                else:
                    enhance_area.append(demand)
            self.enhance_area=np.array(enhance_area).reshape(-1,7)
            
            weaken_area=[]
            for demand in demands[object_name]['weaken']:
                if isinstance(demand,str):
                    for li in self.objects_config[object_name]['pre_defined_feature'][demand]:
                        weaken_area.append(li)
                else:
                    weaken_area.append(demand)
            self.weaken_area=np.array(weaken_area).reshape(-1,7)
            print(f'object name: {object_name} enhance {self.enhance_area.shape[0]} weaken {self.weaken_area.shape[0]}')
            contact_map_value = self.calcContactArea()
            if if_post_process:
                contact_map_value = pre_process_sharp_clamp_np(contact_map_value)
            
            # print('object_point_cloud',object_point_cloud.shape,'contact_map_value',contact_map_value.shape)
            # hist, bin_edges=np.histogram(contact_map_value.cpu().numpy(),bins=100)
            # print('hist, bin_edges',hist, bin_edges)
            # contact_map_value.shape (2048,1)

            cmap_sample['object_point_cloud'] = torch.tensor(self.object_point_cloud)
            cmap_sample['contact_map_value'] = torch.tensor(contact_map_value)
            cmap.append(cmap_sample)
            
            vis_data = []
            contact_map_goal = np.concatenate([self.object_point_cloud, contact_map_value], axis=1)
            vis_data += [plot_point_cloud_cmap(contact_map_goal[:, :3],
                                            contact_map_goal[:, 6])]
            vis_data += [plot_mesh_from_name(f'{object_name}',mesh_path=self.objects_config[object_name]["mesh_file"])]
            fig = go.Figure(data=vis_data)
            fig.write_html(os.path.join(vis_dir, f'{object_name}.html'))
            fig.write_image(file=os.path.join(vis_dir, f'{object_name}.svg'),format='svg')
        torch.save(cmap, cmap_path)
        return cmap


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pre_process', default='sharp_lift', type=str)

    parser.add_argument('--s_model', default='PointNetCVAE_SqrtFullRobots', type=str)

    parser.add_argument('--num_per_object', default=1, type=int)

    parser.add_argument('--comment', default='test_{}'.format(time.strftime('%m_%d_%H_%M_%S', time.localtime())), type=str)
    args_ = parser.parse_args()
    tag = str(time.time())
    return args_, tag


def pre_process_sharp_clamp(contact_map):
    gap_th = 0.5  # delta_th = (1 - gap_th)
    gap_th = min(contact_map.max().item(), gap_th)
    delta_th = (1 - gap_th)
    contact_map[contact_map > 0.4] += delta_th
    # contact_map += delta_th
    contact_map = torch.clamp_max(contact_map, 1.)
    return contact_map

def pre_process_sharp_clamp_np(contact_map):
    gap_th = 0.5  # delta_th = (1 - gap_th)
    gap_th = min(contact_map.max().item(), gap_th)
    delta_th = (1 - gap_th)
    contact_map[contact_map > 0.4] += delta_th
    # contact_map += delta_th
    contact_map = np.clip(contact_map, -1, 1.)
    return contact_map

def identity_map(contact_map):
    return contact_map


if __name__ == '__main__':
    set_global_seed(seed=42)
    args, time_tag = get_parser()

    pre_process_map = {'sharp_lift': pre_process_sharp_clamp,
                       'identity': identity_map}
    pre_process_contact_map_goal = pre_process_map[args.pre_process]

    logs_basedir = os.path.join('logs_inf_cvae', f'{args.s_model}', f'{args.pre_process}', f'{args.comment}-{time_tag}')
    vis_id_dir = os.path.join(logs_basedir, 'vis_id_dir')
    vis_ood_dir = os.path.join(logs_basedir, 'vis_ood_dir')
    vis_dir = os.path.join(logs_basedir, 'vis_dir')
    cmap_path_id = os.path.join(logs_basedir, 'cmap_id.pt')
    cmap_path_ood = os.path.join(logs_basedir, 'cmap_ood.pt')
    cmap_path = os.path.join(logs_basedir, 'cmap.pt')
    
    os.makedirs(logs_basedir, exist_ok=False)
    # os.makedirs(vis_id_dir, exist_ok=False)
    # os.makedirs(vis_ood_dir, exist_ok=False)
    os.makedirs(vis_dir, exist_ok=False)
    
    device = "cuda"

    if args.s_model == 'PointNetCVAE_SqrtFullRobots':
        model_basedir = os.path.join(os.path.dirname(point2grasp.__file__),'../ckpts/SqrtFullRobots')
        from ckpts.SqrtFullRobots.src.utils_model.PointNetCVAE import PointNetCVAE
        model: nn.Module
        model = PointNetCVAE(latent_size=128,
                             encoder_layers_size=[4, 64, 128, 512],
                             decoder_global_feat_size=512,
                             decoder_pointwise_layers_size=[3, 64, 64],
                             decoder_global_layers_size=[64, 128, 512],
                             decoder_decoder_layers_size=[64 + 512 + 128, 512, 64, 64, 1])
        model.load_state_dict(torch.load(os.path.join(model_basedir, 'weights', 'pointnet_cvae_model.pth')))
        model = model.to(device)
        model.eval()
        seen_object_list = json.load(open(os.path.join(model_basedir, "split_train_validate_objects.json"), 'rb'))['train']
        unseen_object_list = json.load(open(os.path.join(model_basedir, "split_train_validate_objects.json"), 'rb'))['validate']
    else:
        raise NotImplementedError("Occur when load model...")

    # focus_part_area={
    #     "contactdb+alarm_clock":[{'x':,'y':,'z':,'r':}]
    # }
    objects_list=[]
    objects_list.append(seen_object_list[0])
    p2pc=Points2PointCloud(objects_config=objects_config,device=device)
    p2pc.genPointCloud(objects_list=objects_list,
                       demands=demands_test,
                       cmap_path=cmap_path,
                       vis_dir=vis_dir)