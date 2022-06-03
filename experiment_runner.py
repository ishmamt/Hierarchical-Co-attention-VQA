import os
import pickle
import torch
from torch.utils.data import DataLoader
from torch.cuda import get_device_name, current_device
from vqa_dataset import VQADataSet
from model import CoattentionNet


class ExperimentRunner():
    ''' Base class for runnung the experiment.
    '''

    def __init__(self, train_images_dir, train_questions_dir, train_annotations_dir,
                val_images_dir, val_questions_dir, val_annotations_dir , batch_size,
                num_of_epochs, num_of_workers, collate, train_results_dir, val_results_dir, 
                saving_frequency, learning_rate, save_results=False):
        ''' Initializes the experiment runner class.
        Parameters:
            train_images_dir: string; Path to train images.
            train_questions_dir: string; Path to train questions.
            train_annotations_dir: string; Path to train annotations.
            val_images_dir: string; Path to validation images.
            val_questions_dir: string; Path to validation questions.
            val_annotations_dir: string; Path to validation annotations.
            batch_size: int; Batch size.
            num_of_epochs: int; Number of epochs.
            num_of_workers: int; Number of workers.
            collate: boolean; Flag to indicate that the results have already been preprocessed and saved. We will not need to do it again. Also images have been encoded.
            train_results_dir: string; Path to the saved train results.
            val_results_dir: string; Path to the saved validation results.
            saving_frequency: int; Frequency of saving in checkpoints.
            learning_rate: float; The learning rate of the algorithm.
            save_results: boolean; Flag for saving results such as vocabulary, top 1000 answers etc.
        '''

        train_dataset = VQADataSet("train", train_questions_dir, train_annotations_dir, train_images_dir, 
                                  "COCO_train2014_", collate, save_results, train_results_dir)
        val_dataset = VQADataSet("val", val_questions_dir, val_annotations_dir, val_images_dir, 
                                "COCO_val2014_", collate, save_results, val_results_dir)

        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                          num_workers=5, collate_fn=self.custom_collate_func)
        self.val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                                        num_workers=5, collate_fn=self.custom_collate_func)

        self.num_of_epochs = num_of_epochs
        self.num_of_workers = num_of_workers
        self.logging_frquency = 10  # every 10 steps
        self.verbose = 50 # every 50 steps
        self.batch_size = batch_size
        self.saving_frequency = saving_frequency  # epoch-wise
        self.learning_rate = learning_rate

        # Loading the model
        with open(os.path.join(self.train_results_dir, "question_vocabulary.pkl"), 'rb') as f:
            question_vocabulary = pickle.load(f)
            
        self.model = CoattentionNet(len(question_vocabulary))
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        if self.DEVICE == "cuda":
            print(f"Loading model with: {self.DEVICE} | {get_device_name(current_device())}.")
            self.model = self.model.cuda()

        # Setting the optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-8)

        # Setting the loss function
        self.criterion = torch.nn.CrossEntropyLoss()
    

    def custom_collate_func(self, seq_list):
        ''' Custome collate function to stack multiple images/questions/answers in a batch.
        Parameters:
            seq_list: *args; Whatever the dataloader returns.
        Returns:
            image_tensors: pytorch tensor object; Stacked image tensor.
            question_tensors: pytorch tensor object; Stacked question tensor.
            answer_tensors: pytorch tensor object; Stacked answer tensor.
        '''

        image_tensors, question_tensors, answer_tensors = zip(*seq_list)  # will recieve a batch of data points.
        lens = [len(question) for question in question_tensors]
        seq_order = sorted(range(len(lens)), key=lens.__getitem__, reverse=True)

        image_tensors = torch.stack([image_tensors[i] for i in seq_order])
        question_tensors = [question_tensors[i] for i in seq_order]
        answer_tensors = torch.stack([answer_tensors[i] for i in seq_order])

        return image_tensors, question_tensors, answer_tensors



if __name__ == "__main__":
    train_image_dir = "Data/train/images/train10K"
    train_qjson = "Data/train/questions/train_quest_10K.json"
    train_ajson = "Data/train/annotations/train_ann_10K.json"
    collate = True
    train_results_dir = "Data/train/cache/"

    val_image_dir = "Data/val/images/val3K"
    val_qjson = "Data/val/questions/val_quest_3K.json"
    val_ajson = "Data/val/annotations/val_ann_3K.json"
    collate = True
    val_results_dir = "Data/val/cache/"


    exp_runner = ExperimentRunner(train_image_dir, train_qjson, train_ajson, val_image_dir, val_qjson, val_ajson, 100, 10, 10, collate, train_results_dir, val_results_dir)
