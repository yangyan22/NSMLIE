import os
import numpy as np
import glob
import cv2
import lpips
from PIL import Image
from data import *
import lpips


def ssim(prediction, target):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    img1 = prediction.astype(np.float64)
    img2 = target.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5] 
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(target, ref):
    '''
    calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    img1 = np.array(target, dtype=np.float64)
    img2 = np.array(ref, dtype=np.float64)
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def calculate_psnr(target, ref):
    img1 = np.array(target, dtype=np.float32)
    img2 = np.array(ref, dtype=np.float32)
    diff = img1 - img2
    psnr = 10.0 * np.log10(255.0 * 255.0 / np.mean(np.square(diff)))
    return psnr

def metrics(im_pattern, label_dir, loss_fn):

    avg_psnr = 0
    avg_ssim = 0
    avg_lpips = 0
    n = 0

    if 'sice' in im_pattern.lower():
        testset = 'SICE'
    elif 'lolv2r' in im_pattern.lower():
        testset = 'LOLv2r'
    else:
        testset = 'LOLv1'
    
    image_paths = sorted(glob.glob(im_pattern))

    for item in image_paths:
        im1 = Image.open(item).convert('RGB')

        if testset == 'LOLv1':
            name = os.path.basename(item)

        elif testset == 'SICE':
            name = os.path.basename(item).split('_')[0] + '.JPG'

        elif testset == 'LOLv2r':
            base_name = os.path.basename(item)
            number_part = os.path.splitext(base_name)[0][-5:]
            name = 'normal' + number_part + '.png'

        else:
            raise ValueError(f"Unsupported testset: {testset}")

        gt_path = os.path.join(label_dir, name)
        if not os.path.exists(gt_path):
            print(f"Warning: GT not found for {item}")
            continue

        im2 = Image.open(gt_path).convert('RGB')

        (w, h) = im2.size
        im1 = im1.resize((w, h))
        im1_np = np.array(im1)
        im2_np = np.array(im2)

        score_psnr = calculate_psnr(im1_np, im2_np)
        score_ssim = calculate_ssim(im1_np, im2_np)

        ref_img = lpips.im2tensor(lpips.load_image(gt_path)).cuda()
        pred_img = lpips.im2tensor(cv2.resize(lpips.load_image(item), (w, h))).cuda()
        score_lpips = loss_fn(ref_img, pred_img).item()

        avg_psnr += score_psnr
        avg_ssim += score_ssim
        avg_lpips += score_lpips
        n += 1

        print(f"[{testset}] {item} - SSIM: {score_ssim:.4f}")

    if n == 0:
        raise RuntimeError(f"No images processed for pattern {im_pattern}")

    avg_psnr /= n
    avg_ssim /= n
    avg_lpips /= n

    return avg_psnr, avg_ssim, avg_lpips

if __name__ == '__main__' :

    loss_fn = lpips.LPIPS(net='alex').cuda()
    im_dir = 'results/LOLv2r_test/R/*'
    label_dir = 'dataset/test/LOLv2r_test/high'

    avg_psnr, avg_ssim, avg_lpips = metrics(im_dir, label_dir,loss_fn)
    print("===> Avg.PSNR: {:.4f} dB ".format(avg_psnr))
    print("===> Avg.SSIM: {:.4f} ".format(avg_ssim))
    print("===> Avg.LPIPS: {:.4f} ".format(avg_lpips))