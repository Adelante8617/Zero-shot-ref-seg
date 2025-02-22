import warnings
warnings.filterwarnings("ignore")

import os
import torch
import cv2
from groundingdino.util.inference import load_model, load_image, predict, annotate
import supervision as sv

# Check PyTorch version and GPU availability
check_env = False
if check_env:
    print(torch.__version__)
    print(torch.cuda.is_available())

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

HOME = os.getcwd()

CONFIG_PATH = os.path.join(HOME, "groundingdino/config/GroundingDINO_SwinT_OGC.py")
WEIGHTS_NAME = "groundingdino_swint_ogc.pth"
WEIGHTS_PATH = os.path.join(HOME, "weights", WEIGHTS_NAME)

if check_env:
    print(CONFIG_PATH, "; exist:", os.path.isfile(CONFIG_PATH))
    print(WEIGHTS_PATH, "; exist:", os.path.isfile(WEIGHTS_PATH))

model = load_model(CONFIG_PATH, WEIGHTS_PATH)

def getBoxFromText(IMAGE_PATH, TEXT_PROMPT, BOX_TRESHOLD = 0.35, TEXT_TRESHOLD = 0.25,visualize=False, output_path=None):
    image_source, image = load_image(IMAGE_PATH)

    # Predict bounding boxes and annotations
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )

    if visualize:
        # Annotate the image with predicted boxes and phrases
        annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)

        if output_path is None:
            output_path = os.path.join(HOME, "output", "annotated.jpeg")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

        cv2.imwrite(output_path, annotated_frame)
        print(f"Annotated image saved at: {output_path}")

    return boxes

