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


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot_name', default='allegro', type=str)

    parser.add_argument('--dataset', default='SqrtFullRobots', type=str)
    parser.add_argument('--dataset_id', default='SharpClamp_A3', type=str)
    parser.add_argument('--max_iter', default=1000, type=int)
    parser.add_argument('--steps_per_iter', default=1, type=int)
    parser.add_argument('--num_particles', default=16, type=int)
    parser.add_argument('--learning_rate', default=5e-3, type=float)
    parser.add_argument('--init_rand_scale', default=0.5, type=float)

    parser.add_argument('--domain', default='ood', type=str)
    parser.add_argument('--object_id', default=1, type=int)
    parser.add_argument('--energy_func', default='align_dist', type=str)
    parser.add_argument('--comment', type=str)
    parser.add_argument('--cmap_dataset', type=str, default='/home/user/xsj/GenDexGrasp/logs_inf_cvae/PointNetCVAE_SqrtUnseenShadowhand/sharp_lift/example-1714820067.769674')
    parser.add_argument('--test_name',default='{}'.format(time.strftime('%m_%d_%H_%M_%S', time.localtime())))
    

    args_ = parser.parse_args()
    tag = str(time.time())
    return args_, tag


if __name__ == '__main__':
    set_global_seed(seed=42)
    torch.set_printoptions(precision=4, sci_mode=False, edgeitems=8)
    args, time_tag = get_parser()
    print(args)
    print(f'double check....')

    logs_basedir = os.path.join('logs_gen', args.test_name)
    print('logs_basedir',logs_basedir)
    exit()
    tb_dir = os.path.join(logs_basedir, 'tb_dir')
    tra_dir = os.path.join(logs_basedir, 'tra_dir')
    os.makedirs(logs_basedir, exist_ok=True)
    os.makedirs(tra_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(tb_dir)

    f = open(os.path.join(logs_basedir, 'command.txt'), 'w')
    f.write(' '.join(sys.argv))
    f.close()

    src_dir_list = ['utils', 'utils_model', 'utils_data']
    os.makedirs(os.path.join(logs_basedir, 'src'), exist_ok=True)
    for fn in os.listdir('.'):
        if fn[-3:] == '.py':
            shutil.copy(fn, os.path.join(logs_basedir, 'src', fn))
    for src_dir in src_dir_list:
        for fn in os.listdir(f'{src_dir}'):
            os.makedirs(os.path.join(logs_basedir, 'src', f'{src_dir}'), exist_ok=True)
            if fn[-3:] == '.py' or fn[-5:] == '.yaml':
                shutil.copy(os.path.join(f'{src_dir}', fn), os.path.join(logs_basedir, 'src', f'{src_dir}', fn))

    robot_name = args.robot_name
    device = "cuda:2" if torch.cuda.is_available() else "cpu"

    # load cmap inference dataset
    try:
        print('args.cmap_dataset',args.cmap_dataset,'args.domain',args.domain)
        cmap_dataset = torch.load(os.path.join(f'{args.cmap_dataset}', f'cmap_{args.domain}.pt'))
    except FileNotFoundError:
        raise NotImplementedError('occur when load CMap Dataset...')

    # init model
    model = AdamGrasp(robot_name=robot_name, writer=writer, contact_map_goal=None,
                      num_particles=args.num_particles, init_rand_scale=args.init_rand_scale, max_iter=args.max_iter,
                      steps_per_iter=args.steps_per_iter, learning_rate=args.learning_rate, device=device,
                      energy_func_name=args.energy_func)
    object_name_list = []
    if args.domain == 'ood':
        object_name_list = json.load(open(os.path.join(f'dataset/CMapDataset-sqrt_align/split_train_validate_objects.json'), 'rb'))['validate']
    elif args.domain == 'id':
        object_name_list = json.load(open(os.path.join(f'dataset/CMapDataset-sqrt_align/split_train_validate_objects.json'), 'rb'))['train']
    else:
        raise NotImplementedError
    object_name_list.sort()
    adam_object_name = object_name_list[args.object_id]
    for i_data in cmap_dataset:
        object_name = i_data['object_name']
        if object_name != adam_object_name:
            continue
        object_point_cloud = i_data['object_point_cloud']
        i_sample = i_data['i_sample']
        contact_map_value = i_data['contact_map_value']
        running_name = f'{object_name}+{i_sample}'
        contact_map_goal = torch.cat([object_point_cloud, contact_map_value], dim=1).to(device)
        record = model.run_adam(object_name=object_name, contact_map_goal=contact_map_goal, running_name=running_name)
        with torch.no_grad():
            q_tra, energy, steps_per_iter = record
            i_record = {'q_tra': q_tra[:, -1:, :].detach(),
                        'energy': energy,
                        'steps_per_iter': steps_per_iter,
                        'dataset': args.dataset,
                        'object_name': object_name,
                        'i_sample': i_sample}
            torch.save(i_record, os.path.join(tra_dir, f'tra-{object_name}-{i_sample}.pt'))

