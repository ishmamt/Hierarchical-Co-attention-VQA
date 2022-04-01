import os
import numpy as np
import pickle
from collections import defaultdict
from vqa import VQA
from image_dataset import get_image_ids

image_dir = "Data/VQA/train/images/train10K"
image_prefix = "COCO_train2014_"
qjson = "Data/VQA/train/questions/train_quest_10K.json"
ajson = "Data/VQA/train/annotations/train_ann_10K.json"

vqa = VQA(ajson, qjson)

image_names = [file for file in os.listdir(image_dir)]
image_ids = get_image_ids(image_names, image_prefix, "list")

ques_ids = vqa.getQuesIds(image_ids)

q2i = defaultdict(lambda: len(q2i))
pad = q2i["<pad>"]
start = q2i["<sos>"]
end = q2i["<eos>"]
UNK = q2i["<unk>"]

print(q2i)
