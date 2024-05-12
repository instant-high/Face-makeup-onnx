import cv2
import numpy as np
import argparse
from skimage.filters import gaussian

from cp.onnx_model import face_parser

    
def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--image', default='test/116.jpg')
    parse.add_argument('--result', default='test/116_makeup.jpg')
    return parse.parse_args()


def sharpen(img):
    img = img * 1.0
    ##gauss_out = gaussian(img, sigma=5, multichannel=True)
    gauss_out = gaussian(img, sigma=5, channel_axis=True)

    alpha = 1.5
    img_out = (img - gauss_out) * alpha + img

    img_out = img_out / 255.0

    mask_1 = img_out < 0
    mask_2 = img_out > 1

    img_out = img_out * (1 - mask_1)
    img_out = img_out * (1 - mask_2) + mask_2
    img_out = np.clip(img_out, 0, 1)
    img_out = img_out * 255
    return np.array(img_out, dtype=np.uint8)


def makeup(image, parsing, part=17, color=[230, 50, 20]):
    b, g, r = color      #[10, 50, 250]       # [10, 250, 10]
    tar_color = np.zeros_like(image)
    tar_color[:, :, 0] = b
    tar_color[:, :, 1] = g
    tar_color[:, :, 2] = r

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    tar_hsv = cv2.cvtColor(tar_color, cv2.COLOR_BGR2HSV)

    #if part == 12 or part == 13:
    #    image_hsv[:, :, 0:2] = tar_hsv[:, :, 0:2]
    #else:
    #    image_hsv[:, :, 0:1] = tar_hsv[:, :, 0:1]
    
    if part != 17:
        image_hsv[:, :, 0:2] = tar_hsv[:, :, 0:2]
    else:
        image_hsv[:, :, 0:1] = tar_hsv[:, :, 0:1]

    changed = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

    if part == 17:
        changed = sharpen(changed)

    changed[parsing != part] = image[parsing != part]
    return changed


if __name__ == '__main__':

    args = parse_args()

    table = {
        'face' : 1,
        'left_brow' : 2,
        'right_brow' : 3,
        'left_eye' : 4,
        'right_eye' : 5,
        'glasses' : 6,
        'left_ear' : 7,
        'right_ear' : 8,
        'nose' : 10,
        'mouth' : 11,
        'upper_lip': 12,
        'lower_lip': 13,
        'neck' : 14,
        'neck_l' : 15,
        'cloth' : 16,
        'hair': 17,
        'hat' : 18        
    }

    #intensity
    intensity = 0.7
    
    image_path = args.image
    cp = 'cp/face_parser.onnx'

    image = cv2.imread(image_path)
    ori = image.copy()
    
    #image = cv2.resize(image,(512,512))
    
    parsing = face_parser(image)

    parsing = cv2.resize(parsing, image.shape[0:2], interpolation=cv2.INTER_NEAREST)
    
    # [B, G, R]
    colors = [[230, 50, 20], [20, 70, 180], [20, 70, 180]] #, [100,0,200]]
    parts = [table['hair'], table['upper_lip'], table['lower_lip']] #, table['face']]

    for part, color in zip(parts, colors):
        image = makeup(image, parsing, part, color)

    cv2.imshow('image', cv2.resize(ori, (512, 512)))
    
    cv2.imshow('color', cv2.resize(image, (512, 512)))
    image = cv2.addWeighted(image, intensity, ori, 1-intensity, 0)
    
    cv2.imwrite(args.result,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()















