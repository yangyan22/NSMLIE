from __future__ import print_function
import argparse
import os
from torch.utils.data import DataLoader
from net.net import *
from data import get_eval_set
from utils import *
from torchvision import transforms
from thop import profile
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='PyTorch')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--rgb_range', type=int, default=1, help='maximum value of RGB')
# parser.add_argument('--data_test', type=str, default='../dataset/LOL/eval/raw')
# parser.add_argument('--data_test', type=str, default='../dataset/SICE2/test/image')
parser.add_argument('--model', default='./results/NSLIE/epoch_85.pth', help='Pretrained base model')   
parser.add_argument('--output_folder', type=str, default='./results/NSLIE/SICE/')

# 定义测试集列表
test_datasets = {
    'LOL': True,
    'SICE': True,
    'LOL-v2R': True,
    'DICM': False,
    'LIME': False,
    'MEF': False,
    'NPE': False,
    'VV': False
}

opt = parser.parse_args()

def eval(dataset_name, use_raw, model):
    torch.set_grad_enabled(False)
    model.eval()
    print('\nEvaluation:')
    
    # 处理数据集文件构成不一致
    if use_raw:
        data_dir = f'./dataset/test/{dataset_name}/raw/'
    else:
        data_dir = f'./dataset/test/{dataset_name}/'

    output_dir = f'./results/NSLIE/{dataset_name}/'
    test_set = get_eval_set(data_dir)
    testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads,
                                     batch_size=opt.testBatchSize, shuffle=False)

    for batch in testing_data_loader:
        input, name = batch[0], batch[1]
        input = input.cuda()

        print(name)

        illumination, reflectance = model(input)

        # 确保输出目录存在
        os.makedirs(os.path.join(output_dir, 'L'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'R'), exist_ok=True)

        illumination = illumination.cpu()
        reflectance = reflectance.cpu()

        illum_img = transforms.ToPILImage()(illumination.squeeze(0))
        reflect_img = transforms.ToPILImage()(reflectance.squeeze(0))

        illum_img.save(os.path.join(output_dir, 'L', name[0]))
        reflect_img.save(os.path.join(output_dir, 'R', name[0]))

    torch.set_grad_enabled(True)


if __name__ == '__main__' :

    print('===> Building model')
    model = enhance_net().cuda()
    model.load_state_dict(torch.load(opt.model, map_location=lambda storage, loc: storage))
    print('Pre-trained model is loaded.')

    for dataset_name, use_raw in test_datasets:
        eval(dataset_name, model)



