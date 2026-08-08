import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from torch.utils.data import DataLoader
from net.net import enhance_net
import argparse
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler as lrs
from data import get_training_set, get_eval_set
from utils import *
import random
from net.losses import *
from torchvision import transforms
from measure import metrics
from datetime import datetime
import csv
import lpips


# Training settings
parser = argparse.ArgumentParser(description='PyTorch LIE')
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=2, help='Snapshots')
parser.add_argument('--start_iter', type=int, default=1, help='Starting Epoch')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=1e-4')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--decay', type=int, default='50', help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decay factor for step decay')
parser.add_argument('--seed', type=int, default=123456789, help='random seed to use.')
parser.add_argument('--data_train', type=str, default='dataset/train/LOLv1_SICE_LOLv2R')
parser.add_argument('--data_test', type=str, default='dataset/test/')
parser.add_argument('--rgb_range', type=int, default=1, help='maximum value of RGB')
parser.add_argument('--weights', default='weights/', help='Location to save checkpoint models')

opt = parser.parse_args()

transform = transforms.Compose([
    transforms.ToPILImage()
])

lpips_idx = lpips.LPIPS(net='alex').cuda()

def seed_torch(seed=opt.seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
seed_torch()

cudnn.benchmark = True


def train():
    model.train()
    loss_print = 0

    for iteration, batch in enumerate(training_data_loader, 1):
        I = batch[0]
        I = I.cuda()

        L, R = model(I)

        # generate sub-image
        mask1, mask2 = generate_mask_pair(I) 
        input_sub1 = generate_subimages(I, mask1)
        input_sub2 = generate_subimages(I, mask2)     

        # Retinex loss
        L_sub1, R_sub1= model(input_sub1)
        loss1 = rec_loss(input_sub1, L_sub1, R_sub1) # |I-L*R| 
        loss2, loss3 = L_loss(input_sub1, L_sub1) # L0, L_prior
        loss4 = R_loss(L_sub1, R_sub1) # R_prior
        Loss_R = loss1  + loss2 * 0.3 + loss3 * 0.3 + loss4 * 0.1

        # consistency loss 
        # mixup , two images share the same sample mask
        R_output_sub1 = generate_subimages(R, mask1)
        R_output_sub2 = generate_subimages(R, mask2)
        input_sub3 = mixup_two_images(input_sub2 , R_output_sub2.detach()) 

        L_sub2, R_sub2 = model(input_sub2)
        _, R_sub3 = model(input_sub3)

        loss_cr1 = torch.nn.MSELoss()(R_sub1, R_sub2)
        loss_cr2 = torch.nn.MSELoss()(R_sub2, R_sub3)
        R_output_sub1 = R_output_sub1.detach()
        R_output_sub2 = R_output_sub2.detach()

        loss_regular =  torch.nn.MSELoss()((R_sub1 - R_sub2), (R_output_sub1 - R_output_sub2))
        Loss_C = loss_cr1 + loss_cr2 + loss_regular 

        # overall loss 
        loss = Loss_R + Loss_C * 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_print = loss_print + loss.item()

        if iteration % 10 == 0:
            print("===> Epoch[{}]({}/{}): Loss: {:.4f} || Learning rate: lr={}.".format(epoch,
                iteration, len(training_data_loader), loss_print, optimizer.param_groups[0]['lr']))
            
            loss_print = 0


def test(testing_data_loader, dataset_name , epoch):

    torch.set_grad_enabled(False)
    model.eval()

    print(f'\nEvaluation on {dataset_name}:')

    for batch in testing_data_loader:
        with torch.no_grad():
            I, name = batch[0], batch[1]

        I = I.cuda()

        with torch.no_grad():
     
            illumination, reflectance = model(I)

        dataset_output_dir = os.path.join(save_dir, dataset_name)
        os.makedirs(os.path.join(dataset_output_dir, 'L'), exist_ok=True)
        os.makedirs(os.path.join(dataset_output_dir, 'R'), exist_ok=True)

        illumination = illumination.cpu()
        reflectance = reflectance.cpu()

        illum_img = transforms.ToPILImage()(illumination.squeeze(0))
        reflect_img = transforms.ToPILImage()(reflectance.squeeze(0))

        illum_img.save(os.path.join(dataset_output_dir, 'L', name[0]))
        reflect_img.save(os.path.join(dataset_output_dir, 'R', name[0]))


    ext = '*.JPG' if dataset_name.lower().startswith('sice') else '*.png'
    im_dir = os.path.join(save_dir, dataset_name, 'R', ext)
    label_dir = os.path.join(opt.data_test, dataset_name, 'high')
    avg_psnr, avg_ssim, avg_lpips = metrics(im_dir, label_dir, lpips_idx)


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


def unified_test_all_datasets(epoch):
    test_datasets = ['LOLv1_test','SICE_test','LOLv2r_test']
    for dataset_name in test_datasets:
        dataset_path = os.path.join(opt.data_test, dataset_name, 'low')
        test_set = get_eval_set(dataset_path)
        testing_data_loader = DataLoader(
            dataset=test_set,
            num_workers=1,
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
timestamp = datetime.now().strftime('%Y%m%d-%H%M')  
subfolder_name = f"{timestamp}"  
save_dir = os.path.join(opt.weights, subfolder_name)  

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print(f"Directory created: {save_dir}")
else:
    print(f"Directory already exists: {save_dir}")

opt.save_dir = save_dir 

for epoch in range(opt.start_iter, opt.nEpochs + 1):
    train()
    scheduler.step()
    if (epoch+1) % opt.snapshots == 0:
        checkpoint(epoch)
        unified_test_all_datasets(epoch)


