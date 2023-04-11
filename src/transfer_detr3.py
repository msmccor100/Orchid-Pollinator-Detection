
# IN this version, I'm trying to incorporate data augmentation via torchivision transforms
import os
from joblib import dump,load
import roboflow
import supervision
import transformers
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import torch
from transformers import DetrForObjectDetection, DetrImageProcessor
import cv2
import torch
import supervision as sv
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import random
import numpy as np
import albumentations as A
from matplotlib import pyplot as plt





HOME = os.getcwd()
print(HOME)

import torch
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)

print(
    "roboflow:", roboflow.__version__, 
    "; supervision:", supervision.__version__, 
    "; transformers:", transformers.__version__, 
    "; pytorch_lightning:", pl.__version__
)



# settings
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CONFIDENCE_TRESHOLD = 0.5
IOU_TRESHOLD = 0.8


#CHECKPOINT = 'facebook/detr-resnet-50'
CHECKPOINT = "D:\\GATECH\\DeepLearning\\FinalProject\\Code\\detr-resnet-50"

image_processor = DetrImageProcessor.from_pretrained(CHECKPOINT)

"""
Note: If this value passed to DetrForObjectDetection.from_pretrained s not set to a full path, for example if 
CHECKPOINT is set to the original value of "facebook/detr_resnet-50 DetrForObjectDetection.from_pretrained will 
try to download this using the function huggingface_hub.hf_hub_download. This hangs. Experimentally, I found this 
same thing happens on the commandline if you do

  >>> from huggingface_hub import hf_hub_download
  >>> hf_hub_download(repo_id="facebook/detr-resnet-50", filename="config.json")

Searching about this problem online, I found you can just git clone the data
 
    $ git clone https://huggingface.co/facebook/detr-resnet-50
    Cloning into 'detr-resnet-50'...
    remote: Enumerating objects: 71, done.
    remote: Counting objects: 100% (9/9), done.
    remote: Compressing objects: 100% (7/7), done.
    remote: Total 71 (delta 2), reused 9 (delta 2), pack-reused 62
    Unpacking objects: 100% (71/71), 15.48 KiB | 113.00 KiB/s, done.
"""

model = DetrForObjectDetection.from_pretrained(CHECKPOINT)
model.to(DEVICE)

# settings
data_location = "d:\GATECH\DeepLearning\FinalProject\Code\datasets\Pollinators-8"
ANNOTATION_FILE_NAME = "_annotations.coco.json"
TRAIN_DIRECTORY = os.path.join(data_location, "train")
VAL_DIRECTORY = os.path.join(data_location, "valid")
TEST_DIRECTORY = os.path.join(data_location, "test")

transform = A.Compose([
    A.HorizontalFlip(p=1.0),
], bbox_params=A.BboxParams(format='coco', min_area=1024, min_visibility=0.1, label_fields=['labels']))


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(
        self, 
        image_directory_path: str, 
        image_processor, 
        train: bool = True
    ):
        annotation_file_path = os.path.join(image_directory_path, ANNOTATION_FILE_NAME)
        super(CocoDetection, self).__init__(image_directory_path, annotation_file_path)
        self.image_processor = image_processor

    def __getitem__(self, idx):
        images, annotations = super(CocoDetection, self).__getitem__(idx)
        #img_np = np.array(images.getdata()).reshape(images.size[0], images.size[1], 3)
        
        #print(annotations)
        #print([anno['bbox'] for anno in annotations])
        #transformed = transform(image=img_np, bboxes=[anno['bbox'] for anno in annotations], labels=[anno['category_id'] for anno in annotations])
        #images = transformed['image']

        #bboxes = transformed['bboxes']
        #print("#########################", bboxes)


        image_id = self.ids[idx]
        annotations = {'image_id': image_id, 'annotations': annotations}
        encoding = self.image_processor(images=images, annotations=annotations, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]

        return pixel_values, target


TRAIN_DATASET = CocoDetection(
    image_directory_path=TRAIN_DIRECTORY, 
    image_processor=image_processor, 
    train=True)

VAL_DATASET = CocoDetection(
    image_directory_path=VAL_DIRECTORY, 
    image_processor=image_processor, 
    train=False)
TEST_DATASET = CocoDetection(
    image_directory_path=TEST_DIRECTORY, 
    image_processor=image_processor, 
    train=False)

import json
with open("D:\\GATECH\\DeepLearning\\FinalProject\\Pollinators-14_COCO_fullsize\\train\\_annotations.coco.json", "r") as fd:
    _annotations =  json.load(fd)
print(list(_annotations.keys()))
print(_annotations["annotations"][0])


image_ids = TRAIN_DATASET.coco.getImgIds()
id = random.choice(image_ids)
id = 0
image = TRAIN_DATASET.coco.loadImgs(id)[0]
annotations = TRAIN_DATASET.coco.imgToAnns[id]
print("##### annotations", annotations)
print("root",TRAIN_DATASET.root)
image_path = os.path.join(TRAIN_DATASET.root, image['file_name'])
print("image_path", image_path)
print("exists?", os.path.exists(image_path))
image = cv2.imread(image_path)

transformed = transform(image=image, bboxes=[anno['bbox'] for anno in annotations], labels=[anno['category_id'] for anno in annotations])
image = transformed['image']
bboxes = transformed['bboxes']
print([anno['bbox'] for anno in annotations])
print(bboxes)
from copy import deepcopy
annots_copy = deepcopy(annotations)
for i, anno in enumerate(annots_copy):
    anno['bbox'] = bboxes[i]


detections = sv.Detections.from_coco_annotations(coco_annotation=annots_copy)
print(detections)
categories = TRAIN_DATASET.coco.cats
id2label = {k: v['name'] for k,v in categories.items()}
labels = [f"{id2label[class_id]}" for _, _, class_id, _ in detections]
box_annotator = sv.BoxAnnotator()
frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

#print('ground truth')
#%matplotlib inline
cv2.imshow("",frame)
cv2.waitKey(0)

raise 1

"""
pixel_values, target = TRAIN_DATASET.__getitem__(3)

        images, annotations = super(CocoDetection, self).__getitem__(idx)
        #img_np = np.array(images.getdata()).reshape(images.size[0], images.size[1], 3)
        
        #print(annotations)
        #print([anno['bbox'] for anno in annotations])
        #transformed = transform(image=img_np, bboxes=[anno['bbox'] for anno in annotations], labels=[anno['category_id'] for anno in annotations])
        #images = transformed['image']

        #bboxes = transformed['bboxes']
        #print("#########################", bboxes)
"""

print("Number of training examples:", len(TRAIN_DATASET))
print("Number of validation examples:", len(VAL_DATASET))
print("Number of test examples:", len(TEST_DATASET))

categories = TRAIN_DATASET.coco.cats
id2label = {k: v['name'] for k,v in categories.items()}


def collate_fn(batch):
    # DETR authors employ various image sizes during training, making it not possible 
    # to directly batch together images. Hence they pad the images to the biggest 
    # resolution in a given batch, and create a corresponding binary pixel_mask 
    # which indicates which pixels are real/which are padding
    pixel_values = [item[0] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    return {
        'pixel_values': encoding['pixel_values'],
        'pixel_mask': encoding['pixel_mask'],
        'labels': labels
    }

TRAIN_DATALOADER = DataLoader(dataset=TRAIN_DATASET, collate_fn=collate_fn, batch_size=4, shuffle=True)
VAL_DATALOADER = DataLoader(dataset=VAL_DATASET, collate_fn=collate_fn, batch_size=4)
TEST_DATALOADER = DataLoader(dataset=TEST_DATASET, collate_fn=collate_fn, batch_size=4)


class Detr(pl.LightningModule):

    def __init__(self, lr, lr_backbone, weight_decay):
        super().__init__()
        self.model = DetrForObjectDetection.from_pretrained(
            pretrained_model_name_or_path=CHECKPOINT, 
            num_labels=len(id2label),
            ignore_mismatched_sizes=True
        )
        
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay

    def forward(self, pixel_values, pixel_mask):
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

        loss = outputs.loss
        loss_dict = outputs.loss_dict

        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)     
        # logs metrics for each training_step, and the average across the epoch
        self.log("training_loss", loss)
        for k,v in loss_dict.items():
            self.log("train_" + k, v.item())

        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)     
        self.log("validation/loss", loss)
        for k, v in loss_dict.items():
            self.log("validation_" + k, v.item())
            
        return loss

    def configure_optimizers(self):
        # DETR authors decided to use different learning rate for backbone
        # you can learn more about it here: 
        # - https://github.com/facebookresearch/detr/blob/3af9fa878e73b6894ce3596450a8d9b89d918ca9/main.py#L22-L23
        # - https://github.com/facebookresearch/detr/blob/3af9fa878e73b6894ce3596450a8d9b89d918ca9/main.py#L131-L139
        param_dicts = [
            {
                "params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.lr_backbone,
            },
        ]
        return torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)

    def train_dataloader(self):
        return TRAIN_DATALOADER

    def val_dataloader(self):
        return VAL_DATALOADER



model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4)
batch = next(iter(TRAIN_DATALOADER))
outputs = model(pixel_values=batch['pixel_values'], pixel_mask=batch['pixel_mask'])


# Don't want to have to retrain if we've already done it once.
if not os.path.exists(os.path.join(os.getcwd(),"trained_model.joblib")):

    MAX_EPOCHS = 50

    # pytorch_lightning < 2.0.0
    # trainer = Trainer(gpus=1, max_epochs=MAX_EPOCHS, gradient_clip_val=0.1, accumulate_grad_batches=8, log_every_n_steps=5)

    # pytorch_lightning >= 2.0.0
    trainer = Trainer(devices=1, accelerator="gpu", max_epochs=MAX_EPOCHS, gradient_clip_val=0.1, accumulate_grad_batches=8, log_every_n_steps=5)

    trainer.fit(model)

    dump(model, "trained_model.joblib")


model = load("trained_model.joblib").to(DEVICE)

#############################################################################################################################################
# Inference on test data

for i in range(1):
    # utils
    categories = TEST_DATASET.coco.cats
    id2label = {k: v['name'] for k,v in categories.items()}
    box_annotator = sv.BoxAnnotator()

    # select random image
    image_ids = TEST_DATASET.coco.getImgIds()
    image_id = random.choice(image_ids)
    print('Image #{}'.format(image_id))

    # load image and annotatons 
    image = TEST_DATASET.coco.loadImgs(image_id)[0]
    annotations = TEST_DATASET.coco.imgToAnns[image_id]
    image_path = os.path.join(TEST_DATASET.root, image['file_name'])
    image = cv2.imread(image_path)
    print("Processing file", image_path)

    # annotate
    detections = sv.Detections.from_coco_annotations(coco_annotation=annotations)
    labels = [f"{id2label[class_id]}" for _, _, class_id, _ in detections]
    frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

    #print('ground truth')
    #%matplotlib inline
    cv2.imshow("",frame)
    cv2.waitKey(0)
    #sv.show_frame_in_notebook(frame, (16, 16))

    # inference
    with torch.no_grad():

        # load image and predict
        inputs = image_processor(images=image, return_tensors='pt').to(DEVICE)
        outputs = model(**inputs)

        # post-process
        target_sizes = torch.tensor([image.shape[:2]]).to(DEVICE)
        results = image_processor.post_process_object_detection(
            outputs=outputs, 
            threshold=CONFIDENCE_TRESHOLD, 
            target_sizes=target_sizes
        )[0]

    # annotate
    detections = sv.Detections.from_transformers(transformers_results=results).with_nms(threshold=0.5)
    labels = [f"{id2label[class_id]} {confidence:.2f}" for _, confidence, class_id, _ in detections]
    frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

    #%matplotlib inline
    #cv2.imshow("",frame)
    cv2.imwrite(str(i) + ".png", frame)
    #cv2.waitKey(0)



