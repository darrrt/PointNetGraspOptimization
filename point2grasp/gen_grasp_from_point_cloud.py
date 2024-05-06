import argparse
import json
import os.path
import time
import sys
import shutil
from utils_model.AdamGrasp import AdamGrasp
import torch
import plotly.graph_objects as go
from utils.visualize_plotly import plot_point_cloud, plot_point_cloud_cmap, plot_mesh_from_name
from utils.set_seed import set_global_seed
from torch.utils.tensorboard import SummaryWriter
import point2grasp
from point2grasp.utils_hand.suggest_pose2joints import ShadowHandGrounding
import numpy as np 
class PointCloud2StableGrasp():
    def __init__(self) -> None:
        pass
    
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot_name', default='shadowhand', type=str)

    parser.add_argument('--dataset', default='SqrtFullRobots', type=str)
    parser.add_argument('--dataset_id', default='SharpClamp_A3', type=str)
    parser.add_argument('--max_iter', default=200, type=int)
    parser.add_argument('--steps_per_iter', default=1, type=int)
    parser.add_argument('--num_particles', default=32, type=int)
    parser.add_argument('--learning_rate', default=5e-3, type=float)
    parser.add_argument('--init_rand_scale', default=0.5, type=float)

    parser.add_argument('--domain', default='ood', type=str)
    parser.add_argument('--object_id', default=1, type=int)
    parser.add_argument('--energy_func', default='align_dist', type=str)
    parser.add_argument('--comment', type=str)
    parser.add_argument('--cmap_dataset', type=str, default='/home/user/xsj/GenDexGrasp/logs_inf_cvae/PointNetCVAE_SqrtFullRobots/sharp_lift/test_05_06_16_30_05-1714984205.0652566')
    parser.add_argument('--goal_orinented', default=True, type=bool)
    parser.add_argument('--enable_goal_orinented_penalty', default=True, type=bool)
    
    args_ = parser.parse_args()
    tag = str(time.time())
    if args_.enable_goal_orinented_penalty==True:
        args_.energy_func='align_dist_with_joint_range_penalty'
        args_.goal_orinented=True
    return args_, tag


if __name__ == '__main__':
    set_global_seed(seed=42)
    torch.set_printoptions(precision=4, sci_mode=False, edgeitems=8)
    torch.set_default_dtype(torch.float32)
    args, time_tag = get_parser()
    print(args)
    print(f'double check....')

    logs_basedir = os.path.join(os.path.join(os.path.dirname(point2grasp.__file__),'../logs_gen'), f'{args.dataset}-{args.dataset_id}', f'{args.domain}-{args.robot_name}-{args.comment}', f'{args.energy_func}')
    tb_dir = os.path.join(logs_basedir, 'tb_dir')
    tra_dir = os.path.join(logs_basedir, 'tra_dir')
    os.makedirs(logs_basedir, exist_ok=True)
    os.makedirs(tra_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(tb_dir)

    f = open(os.path.join(logs_basedir, 'command.txt'), 'w')
    f.write(' '.join(sys.argv))
    f.write(str(args))
    f.close()
    
    src_dir_list = [
        [os.path.join(os.path.dirname(point2grasp.__file__),'utils'),'utils'],
        [os.path.join(os.path.dirname(point2grasp.__file__),'utils_model'),'utils_model'],
        [os.path.join(os.path.dirname(point2grasp.__file__),'utils_data'),'utils_data']]
    os.makedirs(os.path.join(logs_basedir, 'src'), exist_ok=True)
    for fn in os.listdir('.'):
        if fn[-3:] == '.py':
            shutil.copy(fn, os.path.join(logs_basedir, 'src', fn))
    for src_dir in src_dir_list:
        for fn in os.listdir(f'{src_dir[0]}'):
            os.makedirs(os.path.join(logs_basedir, 'src', f'{src_dir[1]}'), exist_ok=True)
            if fn[-3:] == '.py' or fn[-5:] == '.yaml':
                shutil.copy(os.path.join(f'{src_dir[0]}', fn), os.path.join(logs_basedir, 'src', f'{src_dir[1]}', fn))

    robot_name = args.robot_name
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # device = 'cpu'
    # load cmap inference dataset
    try:
        print('args.cmap_dataset',args.cmap_dataset,'args.domain',args.domain)
        cmap_dataset = torch.load(os.path.join(f'{args.cmap_dataset}', f'cmap.pt'))
    except FileNotFoundError:
        raise NotImplementedError('occur when load CMap Dataset...')

    # init model
    model = AdamGrasp(robot_name=robot_name, writer=writer, contact_map_goal=None,
                      num_particles=args.num_particles, init_rand_scale=args.init_rand_scale, max_iter=args.max_iter,
                      steps_per_iter=args.steps_per_iter, learning_rate=args.learning_rate, device=device,
                      energy_func_name=args.energy_func)
    handGrounding = ShadowHandGrounding()
    for i_data in cmap_dataset:
        object_name = i_data['object_name']
        print('object_name',object_name,i_data.keys())
        running_name = f'{object_name}'
        object_point_cloud = i_data['object_point_cloud']
        contact_map_value = i_data['contact_map_value']
        contact_map_goal = torch.cat([object_point_cloud, contact_map_value], dim=1).to(device)
        
        if args.goal_orinented:
            grasp_suggestion_palm_ori_tensor=i_data['grasp_suggestion_palm_ori']
            # grasp_suggestion_palm_ori_tensor[:]=torch.tensor([0,90,90])
            grasp_suggestion_finger_joints_tensor=i_data['grasp_suggestion_finger_joints']
            suggest_params=torch.cat((grasp_suggestion_palm_ori_tensor,grasp_suggestion_finger_joints_tensor)).view(-1)
            print('grasp_suggestion_palm_ori_list,grasp_suggestion_finger_joints_list',suggest_params)
            q_rot_suggest,q_joint_suggest=handGrounding.mapping(suggest_params)
            record = model.run_adam(object_name=object_name, contact_map_goal=contact_map_goal, running_name=running_name,q_joint_suggest=torch.tensor(q_joint_suggest).view(1,-1),q_rot_suggest=torch.tensor(q_rot_suggest).view(1,-1))
        else:
            record = model.run_adam(object_name=object_name, contact_map_goal=contact_map_goal, running_name=running_name)
        with torch.no_grad():
            q_tra, energy, steps_per_iter = record
            i_record = {'q_tra': q_tra[:, -1:, :].detach(),
                        'energy': energy,
                        'steps_per_iter': steps_per_iter,
                        'dataset': args.dataset,
                        'object_name': object_name,
                        'goal_orinented':args.goal_orinented,
                        }
            print(q_tra.shape)
            best_idx=torch.argmin(energy)
            print('np.argmin(energy),np.min(energy)',best_idx,torch.min(energy))
            torch.save(i_record, os.path.join(tra_dir, f'tra-{object_name}.pt'))
            model.opt_model.savefig(os.path.join(logs_basedir, f'tra-{object_name}'),mesh_file_path=i_data['mesh_file'],bold_q=q_tra[best_idx])

