import warnings
warnings.filterwarnings("ignore")

import os
import torch

from groundingdino.util.inference import load_model, load_image, predict, annotate
import supervision as sv
from torchvision.ops import box_convert

import requests
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import warnings
warnings.filterwarnings("ignore", message=".*MultiScaleDeformableAttention.*")

import os
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"

model_id = "IDEA-Research/grounding-dino-tiny"
device = "cuda"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)


def getBoxFromText(IMAGE_PATH, TEXT_PROMPT, BOX_TRESHOLD = 0.25, TEXT_TRESHOLD = 0.2,visualize=False, output_path=None):
    image = Image.open(IMAGE_PATH)

    inputs = processor(images=image, text=[TEXT_PROMPT], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD,
        target_sizes=[image.size[::-1]]
    )

    result = results[0]
    digit_boxes = []
    for box, score, labels in zip(result["boxes"], result["scores"], result["labels"]):
        box = [round(x, 2) for x in box.tolist()]
        digit_boxes.append(box)
        #print(f"Detected {labels} with confidence {round(score.item(), 3)} at location {box}")
    
    return digit_boxes




















#------------------------------
if False:

    # Check PyTorch version and GPU availability
    check_env = False
    if check_env:
        print(torch.__version__)
        print(torch.cuda.is_available())

    DEVICE = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

    HOME = os.getcwd()

    CONFIG_PATH = r"D:/Zero-shot-ref-seg/ObjDetect/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    WEIGHTS_NAME = "groundingdino_swint_ogc.pth"
    WEIGHTS_PATH = r"D:/Zero-shot-ref-seg/ObjDetect/weights/groundingdino_swint_ogc.pth"

    if check_env:
        print(CONFIG_PATH, "; exist:", os.path.isfile(CONFIG_PATH))
        print(WEIGHTS_PATH, "; exist:", os.path.isfile(WEIGHTS_PATH))

    model = load_model(CONFIG_PATH, WEIGHTS_PATH)

    def getBoxFromText(IMAGE_PATH, TEXT_PROMPT, BOX_TRESHOLD = 0.25, TEXT_TRESHOLD = 0.2,visualize=False, output_path=None):
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
            boxes_to_vis = boxes
            # Annotate the image with predicted boxes and phrases
            annotated_frame = annotate(image_source=image_source, boxes=boxes_to_vis, logits=logits, phrases=phrases)

            if output_path is None:
                output_path = os.path.join(HOME, "output", "annotated.jpeg")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

            cv2.imwrite(output_path, annotated_frame)
            print(f"Annotated image saved at: {output_path}")

        # convert GroundingDINO output to xyxy-shape box
        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

        return xyxy

