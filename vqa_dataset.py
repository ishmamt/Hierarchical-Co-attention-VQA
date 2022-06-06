import os
import numpy as np
import errno
import operator
import pickle
from collections import defaultdict
from statistics import mode
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from vqa import VQA
from image_dataset import get_image_ids, get_image_names, load_image


class VQADataSet(Dataset):
    ''' Class for the VQA data set. Uses the VQA python API (https://github.com/GT-Vision-Lab/VQA).
    '''

    def __init__(self, name, questions_json, annotations_json, image_dir, 
                image_prefix, collate=True, save_results=False, results_dir=None):
        ''' Constructor for VQADataSet.
        Parameters:
            name: string; Name of the dataset type (train/val).
            questions_json: string; JSON file for the questions.
            annotations_json: string; JSON file for the annotations.
            image_dir: string; Image directory.
            image_prefix: string; Prefix of image names i.e. "COCO_train2014_".
            collate: boolean; Flag to indicate that the results have already been preprocessed and saved. We will not need to do it again. Also images have been encoded.
            save_results: boolean; Flag for saving results such as vocabulary, top 1000 answers etc.
            results_dir: string; Path to the saved results.
        '''

        self.name = name
        self.questions_json = questions_json
        self.annotations_json = annotations_json
        self.image_dir = image_dir
        self.image_prefix = image_prefix
        self.results_dir = results_dir
        self.save_results = save_results
        self.collate = collate

        if not os.path.exists(self.annotations_json):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.annotations_json)
        if not os.path.exists(self.questions_json):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.questions_json)

        self.vqa = VQA(self.annotations_json, self.questions_json)  # Using the VQA API to load data from the JSON files.
        self.transform = transforms.Compose([transforms.Resize((448, 448)), transforms.ToTensor()])

        if self.collate:
            # if collate is true, then results have already been saved.
            print(f"Loading pre-preocessed files for {self.name}...")
            if not self.results_dir and os.path.isdir(self.results_dir):
                print(self.results_dir)
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.results_dir)

            if self.name == "train":
                with open(os.path.join(self.results_dir, "question_vocabulary.pkl"), 'rb') as f:
                    self.question_vocabulary = pickle.load(f)
                with open(os.path.join(self.results_dir, "top_answers.pkl"), 'rb') as f:
                    self.top_answers = pickle.load(f)
                with open(os.path.join(self.results_dir, "answers_frequency.pkl"), 'rb') as f:
                    self.answers_frequency = pickle.load(f)
            
            elif self.name == "val":
                with open(os.path.join("Data", "train", "cache", "question_vocabulary.pkl"), 'rb') as f:
                    self.question_vocabulary = pickle.load(f)
                with open(os.path.join("Data", "train", "cache", "top_answers.pkl"), 'rb') as f:
                    self.top_answers = pickle.load(f)
                with open(os.path.join("Data", "train", "cache", "answers_frequency.pkl"), 'rb') as f:
                    self.answers_frequency = pickle.load(f)

            self.image_names = np.load(os.path.join(self.results_dir, f"{self.name}_image_names.npy"), encoding='latin1').tolist()
            self.image_ids = np.load(os.path.join(self.results_dir, f"{self.name}_image_ids.npy"), encoding='latin1').tolist()
            self.question_ids = np.load(os.path.join(self.results_dir, f"{self.name}_quest_ids.npy"), encoding='latin1').tolist()

        else:
            if self.name == "train":
                self.question_vocabulary, self.top_answers, self.answers_frequency, self.image_names, self.image_ids, self.question_ids = self.preprocess_dataset()

            elif self.name == "val":
                for i in ["question_vocabulary", "top_answers", "answers_frequency"]:
                    if not os.path.exists(os.path.join("Data", "train", "cache", f"{i}.pkl")):
                        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), os.path.join("Data", "train", "cache", f"{i}.pkl"))

                with open(os.path.join("Data", "train", "cache", "question_vocabulary.pkl"), 'rb') as f:
                    self.question_vocabulary = pickle.load(f)
                with open(os.path.join("Data", "train", "cache", "top_answers.pkl"), 'rb') as f:
                    self.top_answers = pickle.load(f)
                with open(os.path.join("Data", "train", "cache", "answers_frequency.pkl"), 'rb') as f:
                    self.answers_frequency = pickle.load(f)

                self.image_names, self.image_ids, self.question_ids = self.preprocess_dataset()
    

    def __len__(self):
        ''' Overloaded function to return the length of the dataset.
        Returns:
            length: int; Length of the dataset.
        '''

        return len(self.question_ids)
    

    def __getitem__(self, idx):
        ''' Overloaded function to return a single instance from the dataset.
        Parameters:
            idx: int; Index of the instance that we wan to fetch.
        Returns:
            instance: tuple; Returns the image, question and answer.
        '''
        
        question_id = self.question_ids[idx]
        image_id = self.vqa.getImgIds([question_id])[0]

        answers = self.vqa.loadQA(question_id)[0]["answers"]
        question = self.vqa.loadQQA(question_id)[0]["question"][:-1]
        image_name = self.image_names[self.image_ids.index(image_id)]

        # Loading and preparing the image
        image = load_image(os.path.join(self.image_dir, image_name))
        image = self.transform(image).float()

        # Preparing the question
        question = self.encode_question(question)
        question = torch.from_numpy(np.array(question)).long()  # set to int64

        # Preparing the answer. We will take the answer that appears the most.
        answers_list = [answer["answer"].lower() for answer in answers]
        answer = None

        while len(answers_list) > 0:
            if mode(answers_list) in self.top_answers:
                answer = mode(answers_list)
                break
            else:
                answers_list = list(filter(lambda a: a != mode(answers_list), answers_list))  # deleting all instances of that answer from the list.

        if answer:
            answer = torch.from_numpy(np.array([self.top_answers[answer]])).long()
        else:
            answer = torch.from_numpy(np.array([0])).long()  # set to always answer "no" i.e. [0] when encountering something not in the top 1000 answers.

        return image, question, answer
    

    def encode_question(self, question):
        ''' Encode the question using the vocabulary dictionary.
        Parameters:
            question: string; The question that we want to encode.
        Returns:
            encoded_question: list; Encoded Question with <sos> and <eos> at start and end.
        '''

        return [self.question_vocabulary["<sos>"]] + [self.question_vocabulary[x.lower()] for x in question.split(" ") if x.lower() in self.question_vocabulary] + [self.question_vocabulary["<eos>"]]


    def preprocess_dataset(self):
        ''' Preprocessing the VQA dataset.
        '''

        print("Preprocessing VQA Dataset...")
        image_names = get_image_names(self.image_dir)
        image_ids = get_image_ids(image_names, self.image_prefix, "list")
        question_ids = self.vqa.getQuesIds(image_ids)

        if self.name == "train":
            return self.preprocess_dataset_train(image_ids, image_names, 
                                                question_ids)
            
        if self.name == "val":
            if not os.path.exists(os.path.join("Data", "train", "cache", "answers_frequency.pkl")):
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), os.path.join("Data", "train", "cache", "answers_frequency.pkl"))

            with open(os.path.join("Data", "train", "cache", "answers_frequency.pkl"), 'rb') as f:
                answers_frequency = pickle.load(f)

            return self.preprocess_dataset_val(image_ids, image_names, question_ids, answers_frequency)


    def preprocess_dataset_train(self, image_ids, image_names, question_ids):
        ''' Preprocessing training dataset.
        Parameters:
            image_ids: list; List of image ids.
            image_names: list; List of image names.
            question_ids: list; List of question ids.
        Returns:
            question_vocabulary: dict; Vocabulary of questions.
            top_answers: dict; Top 1000 answers.
            answers_frequency: dict; Frequency of answers.
            image_names: list; List of image names.
            image_ids: list; List of image ids.
            question_ids: list; List of question ids.
        '''

        # Creating the question_vocabulary (List of all the words in the questions)
        question_vocabulary = defaultdict(lambda: len(question_vocabulary))
        [question_vocabulary[x] for x in ["<pad>", "<sos>", "<eos>", "<unk>"]]  # We get: {"<pad>": 1, "<sos>":2, "<eos>":3, "<unk>":4}

        # Filling up the question_vocabulary.
        answers_frequency = {}  # {answer: frequency of that answer}
        for question_id in question_ids:
            annotation_dict = self.vqa.loadQA(question_id)[0]
            question_dict = self.vqa.loadQQA(question_id)[0]

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
        
        if self.save_results:
            question_ids_top_answers = []  # question_ids containing answers in the top 1000 answers
            # Fetch question_ids which have answers in the top 1000 answers
            for question_id in question_ids:
                annotation_dict = self.vqa.loadQA(question_id)[0]
                question_dict = self.vqa.loadQQA(question_id)[0]

                not_found = True
                for answer in annotation_dict["answers"]:
                    if answer["answer"].lower() in answers_frequency:
                        not_found = False
                        break

                if not_found:
                    continue
                question_ids_top_answers.append(question_id)
            
            # saving results
            print("Saving pre-processed files...")
            if not self.results_dir:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.results_dir)
            if not os.path.isdir(self.results_dir):
                print(f"Making directory: {self.results_dir}")
                os.makedirs(self.results_dir)

            np.save(os.path.join(self.results_dir, "train_image_names.npy"), image_names)
            np.save(os.path.join(self.results_dir, "train_image_ids.npy"), image_ids)
            np.save(os.path.join(self.results_dir, "train_quest_ids.npy"), question_ids_top_answers)

            with open(os.path.join(self.results_dir, "question_vocabulary.pkl"), 'wb') as f:
                pickle.dump(dict(question_vocabulary), f)
            with open(os.path.join(self.results_dir, "top_answers.pkl"), 'wb') as f:
                pickle.dump(dict(top_answers), f)
            with open(os.path.join(self.results_dir, "answers_frequency.pkl"), 'wb') as f:
                pickle.dump(answers_frequency, f)
    
        return question_vocabulary, top_answers, answers_frequency, image_names, image_ids, question_ids_top_answers


    def preprocess_dataset_val(self, image_ids, image_names, question_ids, answers_frequency):
        ''' Preprocessing validation dataset.
        Parameters:
            image_ids: list; List of image ids.
            image_names: list; List of image names.
            question_ids: list; List of question ids.
        '''

        if self.save_results:
            question_ids_top_answers = []  # question_ids containing answers in the top 1000 answers
            # Fetch question_ids which have answers in the top 1000 answers
            for question_id in question_ids:
                annotation_dict = self.vqa.loadQA(question_id)[0]

                not_found = True
                for answer in annotation_dict["answers"]:
                    if answer["answer"].lower() in answers_frequency:
                        not_found = False
                        break

                if not_found:
                    continue
                question_ids_top_answers.append(question_id)
            
            # saving results
            print("Saving pre-processed files...")
            if not self.results_dir:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.results_dir)
            if not os.path.isdir(self.results_dir):
                print(f"Making directory: {self.results_dir}")
                os.makedirs(self.results_dir)

            np.save(os.path.join(self.results_dir, "val_image_names.npy"), image_names)
            np.save(os.path.join(self.results_dir, "val_image_ids.npy"), image_ids)
            np.save(os.path.join(self.results_dir, "val_quest_ids.npy"), question_ids_top_answers)

            return image_names, image_ids, question_ids_top_answers


if __name__ == "__main__":
    name = "train"
    image_dir = "Data/train/images/train10K"
    image_prefix = "COCO_train2014_"
    qjson = "Data/train/questions/train_quest_10K.json"
    ajson = "Data/train/annotations/train_ann_10K.json"
    collate = True
    save_results = True
    results_dir = "Data/train/cache/"

    train_dataset = VQADataSet(name, qjson, ajson, image_dir, image_prefix, collate, save_results, results_dir)


    name = "val"
    image_dir = "Data/val/images/val3K"
    image_prefix = "COCO_val2014_"
    qjson = "Data/val/questions/val_quest_3K.json"
    ajson = "Data/val/annotations/val_ann_3K.json"
    collate = True
    save_results = True
    results_dir = "Data/val/cache/"

    val_dataset = VQADataSet(name, qjson, ajson, image_dir, image_prefix, collate, save_results, results_dir)