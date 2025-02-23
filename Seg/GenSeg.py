import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

from segment_anything import sam_model_registry, SamPredictor
from functools import reduce

# basic config
# in the future this can be a arg parse config
sam_checkpoint = "D:/Zero-shot-ref-seg/Seg/weight/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cpu"

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
       
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    


def getSegFromBox(image_path, input_boxes:list, visualize=False):

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)
    predictor.set_image(image)

    final_masks = []

    for box in input_boxes:
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box[None,:],
            multimask_output=False,
        )

        onemask = masks[0]
        print(onemask.shape)
        final_masks.append(onemask)

    result = reduce(np.logical_or, final_masks) if len(final_masks) > 0 else None

    if result is None:
        return

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(result, plt.gca())
    for box in input_boxes:
        show_box(box, plt.gca())
    plt.axis('on')
    plt.savefig('output_image.png')  

    if visualize:
        plt.show()

    return result

    

