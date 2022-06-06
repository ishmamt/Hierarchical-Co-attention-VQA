import os
import pickle
from shutil import copyfile
from datetime import datetime
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.cuda import get_device_name, current_device
import torch.nn.utils.rnn as rnn
# from tensorboardX import SummaryWriter
from vqa_dataset import VQADataSet
from model import CoattentionNet
from image_encoder import load_resnet


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

        self.train_dataset = VQADataSet("train", train_questions_dir, train_annotations_dir, train_images_dir, 
                                  "COCO_train2014_", collate, save_results, train_results_dir)
        self.val_dataset = VQADataSet("val", val_questions_dir, val_annotations_dir, val_images_dir, 
                                "COCO_val2014_", collate, save_results, val_results_dir)

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, 
                                          num_workers=num_of_workers, collate_fn=self.custom_collate_func)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, 
                                        num_workers=num_of_workers, collate_fn=self.custom_collate_func)

        self.num_of_epochs = num_of_epochs
        self.num_of_workers = num_of_workers
        self.logging_frquency = 10  # every 10 training steps
        self.verbose = 50 # print every 50 steps
        self.batch_size = batch_size
        self.saving_frequency = saving_frequency  # epoch-wise
        self.learning_rate = learning_rate

        # Loading the model
        with open(os.path.join(train_results_dir, "question_vocabulary.pkl"), 'rb') as f:
            question_vocabulary = pickle.load(f)
            
        self.model = CoattentionNet(len(question_vocabulary), 1000).float()
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        if self.DEVICE == "cuda":
            print(f"Loading model with: {self.DEVICE} | {get_device_name(current_device())}...")
            self.model = self.model.cuda()
        else:
            print(f"Loading model with: {self.DEVICE}...")

        # Setting the optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-8)

        # Setting the loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # Weights initialization
        self.initialize_weights()

        # Logger
        # self.logger = SummaryWriter()  # rewrite this.

        # Image Encoder
        self.image_encoder = load_resnet(self.DEVICE)

        # Checkpoints
        self.checkpoints_dir = os.path.join("saved")
        if not os.path.isdir(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)
        self.experiment_id = len(os.listdir(self.checkpoints_dir))

        if not os.path.isdir(os.path.join(self.checkpoints_dir, str(self.experiment_id))):
            print(f"Creatting {os.path.join(self.checkpoints_dir, str(self.experiment_id))}...")
            os.makedirs(os.path.join(self.checkpoints_dir, str(self.experiment_id)))


    def optimize(self, predicted_answers, true_answers):
        ''' Optimization step for the model.
        Parameters:
            predicted_answers: pytorch tensor object; Tensor of the model predictions.
            true_answers: pytorch tensor object; Ids of ground truth answers.
        Returns:
            loss: float; Loss for the model in current step.
        '''

        self.optimizer.zero_grad()
        loss = self.criterion(predicted_answers, true_answers)
        loss.backward()
        self.optimizer.step()

        return loss
    

    def validate(self):
        ''' Method for validating the model after training.
        Returns:
            accuracy: float; Accuracy of the model after validation.
        '''

        print("Validating the model...")
        accuracy = 0.0
        progress_bar = tqdm(total=len(self.val_dataloader))
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (image_tensors, question_tensors, answer_tensors) in enumerate(self.val_dataloader):
                progress_bar.update(1)
                question_tensors = rnn.pack_sequence(question_tensors)
                image_tensors = image_tensors.to(self.DEVICE)
                image_tensors = self.image_encoder(image_tensors)
                image_tensors = image_tensors.view(image_tensors.size(0), image_tensors.size(1), -1)

                question_tensors = question_tensors.to(self.DEVICE)
                answer_tensors = answer_tensors.to(self.DEVICE)

                answer_tensors = torch.squeeze(answer_tensors)

                # predictions
                predicted_answer = self.model(image_tensors, question_tensors)

                for i in range(answer_tensors.shape[0]):
                    if torch.argmax(predicted_answer[i]).item() == answer_tensors[i]:
                        accuracy += 1.0

                # if (batch_idx + 1) % self.verbose == 0:
                #     print(f"validation accuracy: {round(accuracy / ((batch_idx + 1) * self.batch_size), 2)}")
        
            accuracy = round(accuracy / len(self.val_dataset), 2)

            return accuracy
    

    def train(self):
        ''' Method for training the model.
        '''

        print("\n\n--------------------------------------------------\nTraining the model...\n--------------------------------------------------\n\n")
        train_iteration = 0
        val_iteration = 0
        best_accuracy = 0.0
        loss_history = []
        accuracy_history = []

        for epoch in range(self.num_of_epochs):
            progress_bar = tqdm(total=len(self.train_dataloader))
            self.model.train()
            
            # if (epoch + 1) % 5 == 0:  # HYPER PARAMETER
            #     self.adjust_learning_rate()
            
            for batch_idx, (image_tensors, question_tensors, answer_tensors) in enumerate(self.train_dataloader):
                progress_bar.update(1)

                current_step = epoch * len(self.train_dataloader) + batch_idx

                question_tensors = rnn.pack_sequence(question_tensors)
                image_tensors = image_tensors.to(self.DEVICE)
                image_tensors = self.image_encoder(image_tensors)
                image_tensors = image_tensors.view(image_tensors.size(0), image_tensors.size(1), -1)

                question_tensors = question_tensors.to(self.DEVICE)
                answer_tensors = answer_tensors.to(self.DEVICE)

                answer_tensors = torch.squeeze(answer_tensors)

                # predictions
                predicted_answer = self.model(image_tensors, question_tensors)

                # Optimize the model according to the predictions
                loss = self.optimize(predicted_answer, answer_tensors)
                loss_history.append(loss)

                if (current_step + 1) % self.logging_frquency == 0:
                    # print(f"Epoch: {epoch}, Batch: {batch_idx}/{len(self.train_dataloader)} has loss: {loss}")

                    # self.logger.add_scalar("train/loss", loss.item(), train_iteration)
                    train_iteration += 1

            print(f"Epoch: {epoch} has loss: {loss}")
            
            # Validating
            if (epoch + 1) % self.saving_frequency == 0 or epoch == self.num_of_epochs - 1:
                validation_accuracy = self.validate()
                accuracy_history.append(validation_accuracy)
                print(f"Epoch: {epoch} has validation accuracy {validation_accuracy}")

                # self.logger.add_scalar("valid/accuracy", validation_accuracy, val_iteration)
                val_iteration += 1

                # Remember the best validation accuracy and save a checkpoint
                is_best = validation_accuracy > best_accuracy
                best_accuracy = max(best_accuracy, validation_accuracy)
                self.save_checkpoint(epoch, is_best, best_accuracy)

        # Closing tensorboard logger
        # logger_dir = os.path.join(self.checkpoints_dir, str(self.experiment_id), datetime.now().strftime("%d-%m-%y_%H-%M-%S"))
        # if not os.path.isdir(logger_dir):
        #     os.makedirs(logger_dir)
        # self.logger.export_scalars_to_json(logger_dir + 'tensorboard_summary.json')
        # self.logger.close()


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
    

    def initialize_weights(self):
        ''' Method to initialize model weights.
        '''

        for layer in self.model.modules():
            if isinstance(layer, (torch.nn.Conv1d, torch.nn.Linear)):
                try:
                    torch.nn.init.xavier_normal_(layer.weight)

                    try:
                        torch.nn.init.constant_(layer.bias.data, 0)
                    except:
                        pass
                except:
                    pass
    

    def adjust_learning_rate(self):
        ''' Sets the learning rate to the initial learning rate decayed by 10.
        '''

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] / 10


    def save_checkpoint(self, epoch, is_best, best_accuracy):
        ''' Method for sacving a checkpoint.
        Parameters:
            epoch: int; Epoch number for the checkpoint.
            is_best: boolean; Whether the checkpoint has the best accuracy or not.
            best_accuracy: float; Best accuracy value.
        '''

        state = {"epoch": epoch + 1, 
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_accuracy": best_accuracy
                }

        checkpoint_path = os.path.join(self.checkpoints_dir, str(self.experiment_id), f"{epoch + 1}_checkpoint.pt")
        torch.save(state, checkpoint_path)

        if is_best:
            copyfile(checkpoint_path, os.path.join(self.checkpoints_dir, str(self.experiment_id), "best.pt"))  # create a separate copy of the best checkpoint


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

    saving_frequency = 10
    learning_rate = 0.001
    batch_size = 100
    num_epochs = 2
    num_workers = 5

    exp_runner = ExperimentRunner(train_image_dir, train_qjson, train_ajson, val_image_dir, val_qjson, val_ajson, 
                                batch_size, num_epochs, num_workers, collate, train_results_dir, val_results_dir, saving_frequency, learning_rate)
    
    # exp_runner.save_checkpoint(100, True, 92.4)
    exp_runner.train()
    

    
