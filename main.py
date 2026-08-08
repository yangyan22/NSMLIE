import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from torch.utils.data import DataLoader
from net.net import  enhance_net
import argparse
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler as lrs
from data import get_training_set, get_eval_set
from utils import *
import random
from net.losses import *
from torchvision import transforms
import torchvision.transforms as transforms
from measure import metrics
import shutil
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from logger import Logger
from torch.cuda.amp import autocast, GradScaler
import csv
import lpips


# Training settings
parser = argparse.ArgumentParser(description='PyTorch')
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=2, help='Snapshots')
parser.add_argument('--start_iter', type=int, default=1, help='Starting Epoch')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=1e-4')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--decay', type=int, default='50', help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decay factor for step decay')
parser.add_argument('--seed', type=int, default=123456789, help='random seed to use.')
parser.add_argument('--data_train', type=str, default='dataset/train/LOL_SICE_LOLv2R')
parser.add_argument('--data_test', type=str, default='dataset/test/')
parser.add_argument('--rgb_range', type=int, default=1, help='maximum value of RGB')
parser.add_argument('--weights', default='weights/', help='Location to save checkpoint models')
parser.add_argument('--output_folder', default='results/LOL/', help='Location to save checkpoint models')
parser.add_argument('--log_folder', default='logs.txt', help='Location to save visualizations')
parser.add_argument('--save_dir' ,type=str, help='Directory to save checkpoints')
parser.add_argument('--a', default=0.3,type=float, help='Hyperparameter a')
parser.add_argument('--b', default=0.3,type=float, help='Hyperparameter b')
parser.add_argument('--c', default=0.3,type=float, help='Hyperparameter c')
parser.add_argument('--d', default=0.1,type=float, help='Hyperparameter d')
parser.add_argument('--e', default=1.0,type=float, help='Hyperparameter e')
parser.add_argument('--f', default=1.0,type=float, help='Hyperparameter f')
opt = parser.parse_args()

transform = transforms.Compose([
    transforms.ToPILImage()
])

#加载LPIPS模型
loss_fn = lpips.LPIPS(net='alex').cuda()

def seed_torch(seed=opt.seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
seed_torch()

cudnn.benchmark = True


def train(a,b,c,d,e,f): #abcd分别代表超参数α β λ δ
    model.train()
    loss_print = 0

    for iteration, batch in enumerate(training_data_loader, 1):
        input = batch[0]
        input = input.cuda()

        # 输出中间结果L ，R ，最终结果H
        L, R = model(input)

        # generate sub-image
        mask1, mask2 = generate_mask_pair(input) 
        input_sub1 = generate_subimages(input, mask1)
        input_sub2 = generate_subimages(input, mask2)     

        # Retinex loss
        L_sub1, R_sub1= model(input_sub1)
        # mixup , two images share the same sample mask
        Sub_image3 = generate_subimages(R, mask2)
        input_sub3 = mixup_two_images(input_sub2 , Sub_image3.detach())       
        
        loss1 = rec_loss(input_sub1, L_sub1, R_sub1) # |I-L*R| 
        loss2, loss3 = ill_loss(input_sub1, L_sub1) # L0, KL
        loss4, loss5 = ref_loss(input_sub1, L_sub1, R_sub1) # R0, TR
        LOSS_Retinex = loss1 + loss2 * a + loss3 * b + loss4 * c + loss5 * d 

        # consistency loss 
        L_sub2, R_sub2= model(input_sub2)
        _, R_sub3 = model(input_sub3)

        loss_cr = torch.nn.MSELoss()(R_sub1, R_sub2)
        loss_cr2 = F.mse_loss(R_sub3,R_sub2)
        R_output_sub1 = generate_subimages(R, mask1)
        R_output_sub2 = generate_subimages(R, mask2)
        R_output_sub1 = R_output_sub1.detach()
        R_output_sub2 = R_output_sub2.detach()

        loss_regular =  torch.nn.MSELoss()((R_sub1 - R_sub2), (R_output_sub1 - R_output_sub2))
        LOSS_CR = loss_cr + loss_cr2 + loss_regular 

        # overall loss 
        loss = LOSS_Retinex + LOSS_CR * e

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_print = loss_print + loss.item()

        if iteration % 10 == 0:
            logger.log(
                epoch=epoch,
                iteration=iteration,
                total_iter=len(training_data_loader),
                total_loss=loss.item(),
                loss_dict={
                    'loss1': loss1.item(),
                    'loss2': loss2.item(),
                    'loss3': loss3.item(),
                    'loss4': loss4.item(),
                    'loss5': loss5.item()
                },
                lr=optimizer.param_groups[0]['lr']
            )
            print("===> Epoch[{}]({}/{}): Loss: {:.4f} || Learning rate: lr={}.".format(epoch,
                iteration, len(training_data_loader), loss_print, optimizer.param_groups[0]['lr']))
            
            loss_print = 0


def test(testing_data_loader, dataset_name , epoch):

    torch.set_grad_enabled(False)
    model.eval()

    print(f'\nEvaluation on {dataset_name}:')

    for batch in testing_data_loader:
        with torch.no_grad():
            input, name = batch[0], batch[1]

        input = input.cuda()

        with torch.no_grad():
            # illumination, reflectance, R3, L3 = model(input)
            illumination, reflectance = model(input)
        #中间结果输出
        dataset_output_dir = os.path.join(save_dir, dataset_name)
        os.makedirs(os.path.join(dataset_output_dir, 'L'), exist_ok=True)
        os.makedirs(os.path.join(dataset_output_dir, 'R'), exist_ok=True)
        # os.makedirs(os.path.join(dataset_output_dir, 'R3'), exist_ok=True)

        illumination = illumination.cpu()
        reflectance = reflectance.cpu()
        # R3 = R3.cpu()

        illum_img = transforms.ToPILImage()(illumination.squeeze(0))
        reflect_img = transforms.ToPILImage()(reflectance.squeeze(0))
        # R3_img = transforms.ToPILImage()(R3.squeeze(0))

        illum_img.save(os.path.join(dataset_output_dir, 'L', name[0]))
        reflect_img.save(os.path.join(dataset_output_dir, 'R', name[0]))
        # R3_img.save(os.path.join(dataset_output_dir, 'R3', name[0]))
    
    # 判断文件后缀
    ext = '*.JPG' if dataset_name.lower() == 'sice' else '*.png'

    im_dir = os.path.join(save_dir, dataset_name, 'R', ext)
    label_dir = os.path.join(opt.data_test, dataset_name, 'reference')
    avg_psnr, avg_ssim, avg_lpips = metrics(im_dir, label_dir, loss_fn)

    #将指标写入results.csv
    csv_path = os.path.join(save_dir, f"results-{subfolder_name}-v=0.5.csv")
    write_header = not os.path.exists(csv_path)
    with open(csv_path, mode='a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['epoch', 'dataset', 'psnr', 'ssim', 'lpips'])
        if write_header:
            writer.writeheader()
        writer.writerow({
            'epoch': epoch,
            'dataset': dataset_name,
            'psnr': avg_psnr,
            'ssim': avg_ssim,
            'lpips': avg_lpips
        })

    torch.set_grad_enabled(True)

#调用test测试多个数据集
def unified_test_all_datasets(epoch):
    test_datasets = ['LOL','SICE','LOL-v2R']
    for dataset_name in test_datasets:
        dataset_path = os.path.join(opt.data_test, dataset_name, 'raw')
        test_set = get_eval_set(dataset_path)
        testing_data_loader = DataLoader(
            dataset=test_set,
            num_workers=4,
            batch_size=1,
            shuffle=False
        )
        test(testing_data_loader, dataset_name, epoch)


def checkpoint(epoch):
    if not os.path.exists(opt.weights):
        os.makedirs(opt.weights)
    model_out_path = os.path.join(opt.save_dir,f"epoch_{epoch}.pth")
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")


print('===> Loading datasets')

test_set = get_eval_set(opt.data_test)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)

train_set = get_training_set(opt.data_train)
training_data_loader = DataLoader(
    dataset=train_set, 
    batch_size= opt.batchSize,
    num_workers=opt.threads, 
    shuffle=True,
)

print('===> Building model ')

model= enhance_net().cuda()

optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)

milestones = []
for i in range(1, opt.nEpochs+1):
    if i % opt.decay == 0:
        milestones.append(i)

        
scheduler = lrs.MultiStepLR(optimizer, milestones, opt.gamma)

score_best = 0
a , b , c , d , e , f = opt.a , opt.b , opt.c , opt.d , opt.e , opt.f
timestamp = datetime.now().strftime('%Y%m%d-%H%M')  # 获取当前日期

subfolder_name = f"{timestamp}-a{a}-b{b}-c{c}-d{d}-e{e}-f{f}"  # 生成文件夹名
save_dir = os.path.join(opt.weights, subfolder_name)  # 创建权重保存目录路径

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print(f"Directory created: {save_dir}")
else:
    print(f"Directory already exists: {save_dir}")

opt.save_dir = save_dir 

loss_weights = {'loss1': 1,'loss2': a, 'loss3': b, 'loss4': c, 'loss5': d ,'loss6' : e}
logger = Logger(save_dir=save_dir,model_name='Retinex',
                batch_size= opt.batchSize,
                num_epochs= opt.nEpochs, 
                loss_weights=loss_weights,
                learning_rate= opt.lr, 
                lr_decay=opt.decay,
                gamma= opt.gamma, 
                seed = opt.seed,
                enable_logging=True)


for epoch in range(opt.start_iter, opt.nEpochs + 1):

    train(a,b,c,d,e,f)
    scheduler.step()

    if (epoch+1) % opt.snapshots == 0:
        checkpoint(epoch)
        unified_test_all_datasets(epoch)


