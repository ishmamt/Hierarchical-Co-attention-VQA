import os
import numpy as np
import errno
import operator
import pickle
from collections import defaultdict
from torch.utils.data import Dataset
from vqa import VQA
from image_dataset import get_image_ids, get_image_names, load_image


class VQADataSet(Dataset):
    ''' Class for the VQA data set. Uses the VQA python API (https://github.com/GT-Vision-Lab/VQA)
    '''

    def __init__(self, name, questions_json, annotations_json, image_dir, image_prefix, save_results, results_dir):
        ''' Constructor for VQADataSet.
        Parameters:

        '''

        self.name = name
        self.questions_json = questions_json
        self.annotations_json = annotations_json
        self.image_dir = image_dir
        self.image_prefix = image_prefix
        self.results_dir = results_dir
        self.save_results = save_results

        self.preprocess_dataset(self.save_results, self.results_dir)


    def preprocess_dataset(self, save_results=False, results_dir=None):
        ''' Preprocessing the VQA dataset.
        Parameters:
            save_results: boolean; Flag for saving results.
            results_dir: string; Path to save the results.
        '''

        print("Preprocessing VQA Dataset...")

        if not os.path.exists(self.annotations_json):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.annotations_json)
        if not os.path.exists(self.questions_json):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.questions_json)

        vqa = VQA(self.annotations_json, self.questions_json, )  # Using the VQA API to load data from the JSON files.

        image_names = get_image_names(self.image_dir)
        image_ids = get_image_ids(image_names, self.image_prefix, "list")
        question_ids = vqa.getQuesIds(image_ids)

        if self.name == "train":
            return self.preprocess_dataset_train(vqa, image_ids, image_names, question_ids, save_results, results_dir)
            
        if self.name == "val":
            if not os.path.exists(os.path.join("Data", "train", "cache", "answers_frequency.pkl")):
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), os.path.join("Data", "train", "cache", "answers_frequency.pkl"))

            with open(os.path.join("Data", "train", "cache", "answers_frequency.pkl"), 'rb') as f:
                answers_frequency = pickle.load(f)

            self.preprocess_dataset_val(vqa, image_ids, image_names, question_ids, answers_frequency, save_results, results_dir)


    def preprocess_dataset_train(self, vqa, image_ids, image_names, question_ids, save_results, results_dir):
        # Creating the question_vocabulary (List of all the words in the questions)
        question_vocabulary = defaultdict(lambda: len(question_vocabulary))
        [question_vocabulary[x] for x in ["<pad>", "<sos>", "<eos>", "<unk>"]]  # We get: {"<pad>": 1, "<sos>":2, "<eos>":3, "<unk>":4}

        # Filling up the question_vocabulary.
        answers_frequency = {}  # {answer: frequency of that answer}
        for question_id in question_ids:
            annotation_dict = vqa.loadQA(question_id)[0]
            question_dict = vqa.loadQQA(question_id)[0]

            question = question_dict["question"][:-1]  # Removing the "?"
            [question_vocabulary[x] for x in question.lower().strip().split(" ")]  # Creating the vocabulary

            for answer in annotation_dict["answers"]:
                if not answer["answer_confidence"] == "yes":
                    continue
                if answer["answer"].lower() not in answers_frequency:
                    answers_frequency[answer["answer"].lower()] = 1
                else:
                    answers_frequency[answer["answer"].lower()] = answers_frequency[answer["answer"].lower()] + 1

        top_answers = defaultdict(lambda: len(top_answers))  # {answer: index of the answer sorted by frequency}
        for answer, _ in sorted(answers_frequency.items(), key=operator.itemgetter(1), reverse=True):
            top_answers[answer]
            if len(top_answers) == 1000:
                break  # take only 1000 most frequent answers for classification.
        
        if save_results:
            question_ids_top_answers = []  # question_ids containing answers in the top 1000 answers
            # Fetch question_ids which have answers in the top 1000 answers
            for question_id in question_ids:
                annotation_dict = vqa.loadQA(question_id)[0]
                question_dict = vqa.loadQQA(question_id)[0]

                not_found = True
                for answer in annotation_dict["answers"]:
                    if answer["answer"].lower() in answers_frequency:
                        not_found = False
                        break

                if not_found:
                    continue
                question_ids_top_answers.append(question_id)
            
            # saving results
            print("Saving results...")
            if not results_dir:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), results_dir)
            if not os.path.isdir(results_dir):
                print(f"Making directory: {results_dir}")
                os.mkdir(results_dir)

            np.save(os.path.join(results_dir, "train_image_names.npy"), image_names)
            np.save(os.path.join(results_dir, "train_image_ids.npy"), image_ids)
            np.save(os.path.join(results_dir, "train_quest_ids.npy"), question_ids_top_answers)

            with open(os.path.join(results_dir, "question_vocabulary.pkl"), 'wb') as f:
                pickle.dump(dict(question_vocabulary), f)
            with open(os.path.join(results_dir, "top_answers.pkl"), 'wb') as f:
                pickle.dump(dict(top_answers), f)
            with open(os.path.join(results_dir, "answers_frequency.pkl"), 'wb') as f:
                pickle.dump(answers_frequency, f)
    
        return question_vocabulary, top_answers, answers_frequency


    def preprocess_dataset_val(self, vqa, image_ids, image_names, question_ids, answers_frequency, save_results, results_dir):
        if save_results:
            question_ids_top_answers = []  # question_ids containing answers in the top 1000 answers
            # Fetch question_ids which have answers in the top 1000 answers
            for question_id in question_ids:
                annotation_dict = vqa.loadQA(question_id)[0]

                not_found = True
                for answer in annotation_dict["answers"]:
                    if answer["answer"].lower() in answers_frequency:
                        not_found = False
                        break

                if not_found:
                    continue
                question_ids_top_answers.append(question_id)
            
            # saving results
            print("Saving results...")
            if not results_dir:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), results_dir)
            if not os.path.isdir(results_dir):
                print(f"Making directory: {results_dir}")
                os.mkdir(results_dir)

            np.save(os.path.join(results_dir, "val_image_names.npy"), image_names)
            np.save(os.path.join(results_dir, "val_image_ids.npy"), image_ids)
            np.save(os.path.join(results_dir, "val_quest_ids.npy"), question_ids_top_answers)





if __name__ == "__main__":
    name = "train"
    image_dir = "Data/train/images/train10K"
    image_prefix = "COCO_train2014_"
    qjson = "Data/train/questions/train_quest_10K.json"
    ajson = "Data/train/annotations/train_ann_10K.json"
    save_results = True
    results_dir = "Data/train/cache"

    train_dataset = VQADataSet(name, qjson, ajson, image_dir, image_prefix, save_results, results_dir)


    name = "val"
    image_dir = "Data/val/images/val3K"
    image_prefix = "COCO_val2014_"
    qjson = "Data/val/questions/val_quest_3K.json"
    ajson = "Data/val/annotations/val_ann_3K.json"
    save_results = True
    results_dir = "Data/val/cache"

    val_dataset = VQADataSet(name, qjson, ajson, image_dir, image_prefix, save_results, results_dir)