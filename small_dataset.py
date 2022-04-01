import os
import random
import shutil
import json
from image_dataset import get_image_ids


# num_images = 3000
# image_dir = "Data/VQA/test/images/test2015"
# dest_dir = "Data/VQA/test/images/test3K"

# if not os.path.isdir(dest_dir):
#     print(f"Creating {dest_dir}")
#     os.mkdir(dest_dir)

# images = random.sample(os.listdir(image_dir), num_images)
# counter = 1
# for i in images:
#     src_path = os.path.join(image_dir, i)
#     print(f"{counter}: Copying {i}")
#     shutil.copy(src_path, dest_dir)
#     counter += 1


ann_dir = "Data/VQA/train/annotations/v2_mscoco_train2014_annotations.json"
q_dir = "Data/VQA/train/questions/v2_OpenEnded_mscoco_train2014_questions.json"
image_dir = "Data/VQA/train/images/train10K"
ann_filename = "train_ann_10K.json"
q_filename = "train_quest_10K.json"

image_names = [file for file in os.listdir(image_dir)]
image_ids = get_image_ids(image_names, "COCO_train2014_", "dict")
print("Got image ids")
# print(image_ids)

print("Loading JSONs")
ann = json.load(open(ann_dir, 'r'))
print("Annotation json loaded")
quest = json.load(open(q_dir, 'r'))
print("Question json loaded")

# print(type(ann["annotations"]))
# print(len(ann["annotations"]))
# print(ann["annotations"][238])
# print(ann.keys())

small_ann = list()
small_q = list()
small_ann_json = {}
small_q_json = {}
temp = list()

small_ann_json["info"] = ann["info"]
small_ann_json["license"] = ann["license"]
small_ann_json["data_subtype"] = ann["data_subtype"]
small_ann_json["data_type"] = ann["data_type"]

small_q_json["info"] = quest["info"]
small_q_json["task_type"] = quest["task_type"]
small_q_json["data_type"] = quest["data_type"]
small_q_json["data_subtype"] = quest["data_subtype"]
small_q_json["license"] = quest["license"]

print("Starting annotations loop")
for a in ann["annotations"]:
    if a["image_id"] in image_ids:
        small_ann.append(a)
        temp.append(a["question_id"])

print("Starting questions loop")
for q in quest["questions"]:
    if q["question_id"] in temp:
        small_q.append(q)

small_ann_json["annotations"] = small_ann
small_q_json["questions"] = small_q

print("Dumping JSONs")
small_ann_json = json.dumps(small_ann_json)
print("Annotation JSON dumped")
small_q_json = json.dumps(small_q_json)
print("Question JSON dumped")

with open(ann_filename, "w") as file:
    file.write(small_ann_json)
print("Annotation JSON saved")

with open(q_filename, "w") as file:
    file.write(small_q_json)
print("Question JSON dumped")
