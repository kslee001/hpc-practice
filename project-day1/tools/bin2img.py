from PIL import Image
import sys
import numpy as np
from functools import reduce

imgSize = (1,2,128,191)

def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)

if __name__=='__main__':

  sizeMult = reduce(lambda x, y: x*y, imgSize)
  mask_values = [0,1]
  imgAll = np.fromfile(sys.argv[1], dtype=np.float32)

  for idx in range(int(sys.argv[3])):
    img = imgAll[(sizeMult * idx) : (sizeMult * (idx+1))].reshape(imgSize)
    mask = np.argmax(img, axis=1)

    mask = np.squeeze(mask[0].astype(int))
    result = mask_to_image(mask, mask_values)
    result.save(sys.argv[2].split('.')[0]+'_'+str(idx)+'.png')    