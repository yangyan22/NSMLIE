from __future__ import print_function
import argparse
import os
from torch.utils.data import DataLoader
from net.net import *
from data import get_eval_set
from utils import *
from torchvision import transforms
from thop import profile
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='PyTorch')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--rgb_range', type=int, default=1, help='maximum value of RGB')
parser.add_argument('--model', default='weights/NSMLIE.pth', help='Pretrained base model')   


test_datasets = {
    'LOLv1_test': True,
    'SICE_test': True,
    'LOLv2r_test': True,
}

opt = parser.parse_args()

def eval(dataset_name, model):
    torch.set_grad_enabled(False)
    model.eval()
    print('\nEvaluation:')
    

    data_dir = f'./dataset/test/{dataset_name}/low/'
    output_dir = f'results/{dataset_name}/'
    test_set = get_eval_set(data_dir)
    testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads,
                                     batch_size=opt.testBatchSize, shuffle=False)

    for batch in testing_data_loader:
        input, name = batch[0], batch[1]
        input = input.cuda()

        print(name)

        illumination, reflectance = model(input)


        # input_tensor = torch.randn(1, 3, 256, 256).cuda()
        # flops, params = profile(model, (input_tensor,))
        # print(f"FLOPs: {flops / 1e9:.3f} G")
        # print(f"Params: {params / 1e6:.3f} M")

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

    for dataset_name in test_datasets:
        eval(dataset_name, model)



