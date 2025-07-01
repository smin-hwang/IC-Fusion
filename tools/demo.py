import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from PIL import Image, ImageDraw
import torch
import torchvision.transforms as transforms
from src.zoo import RTDETR 
from src.core import YAMLConfig 

from pytictoc import TicToc

import cv2


# Initialize TicToc
t = TicToc()

# Threshold and configuration
thrh = 0.5
yaml_path = "configs/rtdetr/rtdetr_r50vd_6x_coco.yml"
cfg = YAMLConfig(yaml_path)

# Load pretrained model
model = cfg.model
state = torch.load('./output/rtdetr_r50vd_6x_coco/best_stat.pth', map_location='cpu')
model.load_state_dict(state['model'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

def resize_image(image):
    return transforms.Resize((640, 640))(image)

def pil_to_tensor(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device)

def process_image_sequence(image_folder, output_dir):
    rgb_folder = os.path.join(image_folder, "RGB")
    ir_folder = os.path.join(image_folder, "IR")

    rgb_files = sorted(os.listdir(rgb_folder))
    ir_files = sorted(os.listdir(ir_folder))

    assert len(rgb_files) == len(ir_files), "Mismatch between RGB and IR images"

    for rgb_file, ir_file in zip(rgb_files, ir_files):
        rgb_path = os.path.join(rgb_folder, rgb_file)
        ir_path = os.path.join(ir_folder, ir_file)

        rgb_image = Image.open(rgb_path).convert("RGB")
        ir_image = Image.open(ir_path).convert("RGB")

        # Resize and convert images to tensor
        resized_rgb = resize_image(rgb_image)
        resized_ir = resize_image(ir_image)

        rgb_tensor = pil_to_tensor(resized_rgb)
        ir_tensor = pil_to_tensor(resized_ir)

        t.tic()  # Start timer
        # Perform object detection using the model
        outputs = model(rgb_tensor, ir_tensor)
        t.toc()  # End timer

        orig_target_sizes = torch.tensor(rgb_image.size).to(device)
        results = cfg.postprocessor(outputs, orig_target_sizes)

        # Save image with bounding boxes
        # save_images_with_bboxes(results, output_dir, rgb_path)
        save_images_with_bboxes(results, output_dir, ir_path)

# Function to save images with bounding boxes
def save_images_with_bboxes(results, output_dir, image_path):
    os.makedirs(output_dir, exist_ok=True)
    label_mapping = get_label_mapping()
    image_id = os.path.basename(image_path).split('.')[0]
    image = cv2.imread(image_path)

    results = results[0]
    for bbox, score, label in zip(results["boxes"], results["scores"], results["labels"]):
        if score >= thrh:
            bbox = list(map(int, bbox))
            color = get_color(int(label))
            image = cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            label_name = label_mapping[int(label)]
            text = f"{label_name} {score:.3f}"
            cv2.putText(image, text, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    output_path = os.path.join(output_dir, f"{image_id}_bbox.jpg")
    cv2.imwrite(output_path, image)


def get_color(label):
    color_mapping = {
    1: (203, 192, 255),  
    2: (153, 255, 204),      
    3: (255, 204, 102)     
    }
    return color_mapping.get(label, (0, 0, 0))  

def get_label_mapping():
    label_mapping = {
    1: 'person',
    2: 'car',
    3: 'bicycle'
    }
    return label_mapping


def main():
    image_folder = "./FLIR"
    # output_dir = "./demo_output/FLIR_RGB"
    output_dir = "./demo_output/FLIR_IR"

    process_image_sequence(image_folder, output_dir)

if __name__ == "__main__":
    main()
