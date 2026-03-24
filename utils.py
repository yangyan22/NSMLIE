import os
import torch
import numpy as np
from PIL import Image


def pil_to_np(img_PIL, with_transpose=True):
    """
    Converts image in PIL format to np.array.

    From W x H x C [0...255] to C x W x H [0..1]
    """
    ar = np.array(img_PIL)
    if len(ar.shape) == 3 and ar.shape[-1] == 4:
        ar = ar[:, :, :3]
        # this is alpha channel
    if with_transpose:
        if len(ar.shape) == 3:
            ar = ar.transpose(2, 0, 1)
        else:
            ar = ar[None, ...]

    return ar.astype(np.float32) / 255.


def np_to_pil(img_np):
    """
    Converts image in np.array format to PIL image.

    From C x W x H [0..1] to  W x H x C [0...255]
    :param img_np:
    :return:
    """
    ar = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        assert img_np.shape[0] == 3, img_np.shape
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)


def np_to_torch(img_np):
    """
    Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]

    :param img_np:
    :return:
    """
    return torch.from_numpy(img_np)[None, :]


def torch_to_np(img_var):
    """
    Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    :param img_var:
    :return:
    """
    return img_var.detach().cpu().numpy()[0]

def mixup_two_images(img1, img2):
    lam = np.random.uniform(0.0, 1.0)
    mixed_img = lam * img1 + (1 - lam) * img2
    return mixed_img

operation_seed_counter = 0
def get_generator():
    global operation_seed_counter
    operation_seed_counter += 1
    g_cuda_generator = torch.Generator(device="cuda")
    g_cuda_generator.manual_seed(operation_seed_counter)
    return g_cuda_generator


def space_to_depth(x, block_size):
    n, c, h, w = x.size()
    unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(n, c * block_size**2, h // block_size,
                           w // block_size)


def generate_mask_pair(img):
    # prepare masks (N x C x H/2 x W/2)
    n, c, h, w = img.shape
    mask1 = torch.zeros(size=(n * h // 2 * w // 2 * 4, ),
                        dtype=torch.bool,
                        device=img.device)
    mask2 = torch.zeros(size=(n * h // 2 * w // 2 * 4, ),
                        dtype=torch.bool,
                        device=img.device)
    # prepare random mask pairs
    idx_pair = torch.tensor(
        [
        [0, 1], [0, 2], [1, 0], [2, 0], [0,3], [3,0], [1, 3], [1,2], [3, 1], [2,1], [2, 3], [3, 2]
        ],
        dtype=torch.int64,
        device=img.device)
    rd_idx = torch.zeros(size=(n * h // 2 * w // 2, ),
                         dtype=torch.int64,
                         device=img.device)
    torch.randint(low=0,
                  high=12,
                  size=(n * h // 2 * w // 2, ),
                  generator=get_generator(),
                  out=rd_idx)
    rd_pair_idx = idx_pair[rd_idx]
    rd_pair_idx += torch.arange(start=0,
                                end=n * h // 2 * w // 2 * 4,
                                step=4,
                                dtype=torch.int64,
                                device=img.device).reshape(-1, 1)
    # get masks
    mask1[rd_pair_idx[:, 0]] = 1
    mask2[rd_pair_idx[:, 1]] = 1
    return mask1, mask2


def generate_subimages(img, mask):
    n, c, h, w = img.shape
    subimage = torch.zeros(n,
                           c,
                           h // 2,
                           w // 2,
                           dtype=img.dtype,
                           layout=img.layout,
                           device=img.device)
    # per channel
    for i in range(c):
        img_per_channel = space_to_depth(img[:, i:i + 1, :, :], block_size=2)
        img_per_channel = img_per_channel.permute(0, 2, 3, 1).reshape(-1)
        subimage[:, i:i + 1, :, :] = img_per_channel[mask].reshape(
            n, h // 2, w // 2, 1).permute(0, 3, 1, 2)
    return subimage

class LocalMean(torch.nn.Module):
    '''
    '''
    def __init__(self, patch_size=5):
        super(LocalMean, self).__init__()
        self.patch_size = patch_size
        self.padding = self.patch_size // 2

    def forward(self, image):
        image = torch.nn.functional.pad(image, (self.padding, self.padding, self.padding, self.padding), mode='reflect')
        patches = image.unfold(2, self.patch_size, 1).unfold(3, self.patch_size, 1)
        return patches.mean(dim=(4, 5))

def padr_tensor(img):
    pad=2
    pad_mod=torch.nn.ConstantPad2d(pad,0)
    img_pad=pad_mod(img)
    return img_pad
    
def calculate_local_variance(train_noisy):
    b,c,w,h=train_noisy.shape
    avg_pool = torch.nn.AvgPool2d(kernel_size=5,stride=1,padding=2)
    noisy_avg= avg_pool(train_noisy)
    noisy_avg_pad=padr_tensor(noisy_avg)
    train_noisy=padr_tensor(train_noisy)
    unfolded_noisy_avg=noisy_avg_pad.unfold(2,5,1).unfold(3,5,1)
    unfolded_noisy=train_noisy.unfold(2,5,1).unfold(3,5,1)
    unfolded_noisy_avg=unfolded_noisy_avg.reshape(unfolded_noisy_avg.shape[0],-1,5,5)
    unfolded_noisy=unfolded_noisy.reshape(unfolded_noisy.shape[0],-1,5,5)
    noisy_diff_squared=(unfolded_noisy-unfolded_noisy_avg)**2
    noisy_var=torch.mean(noisy_diff_squared,dim=(2,3))
    noisy_var=noisy_var.view(b,c,w,h)
    return noisy_var

def gauss_cdf(x):
    return 0.5*(1+torch.erf(x/torch.sqrt(torch.tensor(2.))))

def gauss_kernel(kernlen=21,nsig=3,channels=1):
    interval=(2*nsig+1.)/(kernlen)
    x=torch.linspace(-nsig-interval/2.,nsig+interval/2.,kernlen+1,).cuda()
    #kern1d=torch.diff(torch.erf(x/math.sqrt(2.0)))/2.0
    kern1d=torch.diff(gauss_cdf(x))
    kernel_raw=torch.sqrt(torch.outer(kern1d,kern1d))
    kernel=kernel_raw/torch.sum(kernel_raw)
    #out_filter=kernel.unsqueeze(2).unsqueeze(3).repeat(1,1,channels,1)
    out_filter=kernel.view(1,1,kernlen,kernlen)
    out_filter = out_filter.repeat(channels,1,1,1)
    return  out_filter

def blur(x):
    device = x.device
    kernel_size = 21
    padding = kernel_size // 2
    kernel_var = gauss_kernel(kernel_size, 1, x.size(1)).to(device)
    x_padded = torch.nn.functional.pad(x, (padding, padding, padding, padding), mode='reflect')
    return torch.nn.functional .conv2d(x_padded, kernel_var, padding=0, groups=x.size(1))

def pair_downsampler(img):
    # img has shape B C H W
    c = img.shape[1]
    filter1 = torch.FloatTensor([[[[0, 0.5], [0.5, 0]]]]).to(img.device)
    filter1 = filter1.repeat(c, 1, 1, 1)
    filter2 = torch.FloatTensor([[[[0.5, 0], [0, 0.5]]]]).to(img.device)
    filter2 = filter2.repeat(c, 1, 1, 1)
    output1 = torch.nn.functional.conv2d(img, filter1, stride=2, groups=c)
    output2 = torch.nn.functional.conv2d(img, filter2, stride=2, groups=c)
    return output1,output2
