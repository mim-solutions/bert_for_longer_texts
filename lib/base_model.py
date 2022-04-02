import numpy as np
import torch
from torch import nn

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


class Model():
    '''
    Abstract class for models
    '''

    def __init__(self):
        self.params = None
        self.preprocessor = None
        self.dataset_class = None
        self.collate_fn = None

    def evaluate_single_batch(self, batch, model, device):
        raise NotImplementedError("This is implemented for subclasses only")

    def create_dataset(self, X_preprocessed, y):
        dataset = self.dataset_class(X_preprocessed, y)
        return dataset

    def fit(self, X_train, y_train, epochs=3):
        number_of_train_samples = len(X_train)
        # Text preprocessing
        X_train_preprocessed = self.preprocessor.preprocess(X_train)
        # Creating train dataset
        train_dataset = self.create_dataset(X_train_preprocessed, y_train)
        # Creating train dataloader
        train_dataloader = create_train_dataloader(
            train_dataset, self.params['batch_size'], self.collate_fn)
        # Training
        self.train_for_epochs(
            number_of_train_samples,
            train_dataloader,
            epochs)

    def train_and_evaluate(self, X_train, X_test, y_train, y_test, epochs=3):
        number_of_train_samples = len(X_train)
        number_of_test_samples = len(X_test)
        X_train_preprocessed = self.preprocessor.preprocess(X_train)
        X_test_preprocessed = self.preprocessor.preprocess(X_test)
        train_dataset = self.create_dataset(X_train_preprocessed, y_train)
        test_dataset = self.create_dataset(X_test_preprocessed, y_test)
        train_dataloader = create_train_dataloader(
            train_dataset, self.params['batch_size'], self.collate_fn)
        test_dataloader = create_test_dataloader(
            test_dataset, self.params['batch_size'], self.collate_fn)
        result = self.train_and_evaluate_preprocessed(
            number_of_train_samples,
            train_dataloader,
            number_of_test_samples,
            test_dataloader,
            epochs)
        return result

    def train_and_evaluate_preprocessed(
            self,
            number_of_train_samples,
            train_dataloader,
            number_of_test_samples,
            test_dataloader,
            epochs):
        result = {
            'train_acc': [],
            'train_loss': [],
            'test_acc': [],
            'test_loss': []}
        for epoch in range(epochs):
            avg_loss, accuracy = self.train_single_epoch(
                number_of_train_samples, train_dataloader)
            print(
                f'Epoch: {epoch}, Train accuracy: {accuracy}, Train loss: {avg_loss}')
            result['train_acc'].append(accuracy)
            result['train_loss'].append(avg_loss)
            avg_loss, accuracy, _, _ = self.evaluate_single_epoch(
                number_of_test_samples, test_dataloader)
            print(
                f'Epoch: {epoch}, Test accuracy: {accuracy}, Test loss: {avg_loss}')
            result['test_acc'].append(accuracy)
            result['test_loss'].append(avg_loss)
        return result

    def predict(self, X_test):
        test_samples = len(X_test)
        y_test = [0] * len(X_test)  # dummy labels
        X_test_preprocessed = self.preprocessor.preprocess(X_test)
        test_dataset = self.create_dataset(X_test_preprocessed, y_test)
        test_dataloader = create_test_dataloader(
            test_dataset, self.params['batch_size'], self.collate_fn)
        _, _, preds, _ = self.evaluate_single_epoch(
            test_samples, test_dataloader)
        return preds

    def train_single_epoch(self, number_of_train_samples, train_dataloader):
        model = self.nn
        model.train()
        total_loss = 0
        total_accurate = 0

        for _, batch in enumerate(train_dataloader):

            self.optimizer.zero_grad()

            preds, labels = self.evaluate_single_batch(
                batch, model, self.params['device'])

            loss, total_accurate, total_loss = calc_loss_and_accuracy(
                preds, labels, total_loss, total_accurate)

            loss.backward()
            self.optimizer.step()

        avg_loss = total_loss / number_of_train_samples
        accuracy = total_accurate / number_of_train_samples
        return avg_loss, accuracy

    def train_for_epochs(
            self,
            number_of_train_samples,
            train_dataloader,
            epochs=3):
        for epoch in range(epochs):
            avg_loss, accuracy = self.train_single_epoch(
                number_of_train_samples, train_dataloader)
            print(
                f'Epoch: {epoch}, Train accuracy: {accuracy}, Train loss: {avg_loss}')

    def evaluate_single_epoch(self, test_samples, test_dataloader):
        model = self.nn
        total_loss = 0
        total_accurate = 0

        preds_total = []
        labels_total = []

        # deactivate dropout layers
        model.eval()

        for _, batch in enumerate(test_dataloader):

            with torch.no_grad():

                preds, labels = self.evaluate_single_batch(
                    batch, model, self.params['device'])
                preds_total.extend(preds)
                labels_total.extend(labels)

                loss, total_accurate, total_loss = calc_loss_and_accuracy(
                    preds, labels, total_loss, total_accurate)

        # compute the evaluation loss of the epoch
        preds_total = [x.item() for x in preds_total]
        labels_total = [x.item() for x in labels_total]
        avg_loss = total_loss / test_samples
        accuracy = total_accurate / test_samples
        return avg_loss, accuracy, preds_total, labels_total


def create_dataloader(data, sampler_class, batch_size, collate_fn=None):
    sampler = sampler_class(data)
    dataloader = DataLoader(
        data,
        sampler=sampler,
        batch_size=batch_size,
        collate_fn=collate_fn)
    return dataloader


def create_train_dataloader(train_data, batch_size, collate_fn=None):
    train_dataloader = create_dataloader(
        train_data, RandomSampler, batch_size, collate_fn)
    return train_dataloader


def create_test_dataloader(test_data, batch_size, collate_fn=None):
    test_dataloader = create_dataloader(
        test_data, SequentialSampler, batch_size, collate_fn)
    return test_dataloader


def create_dataloaders(train_data, test_data, batch_size, collate_fn=None):

    train_dataloader = create_train_dataloader(
        train_data, batch_size, collate_fn)
    test_dataloader = create_test_dataloader(test_data, batch_size, collate_fn)

    return train_dataloader, test_dataloader


def calc_loss_and_accuracy(preds, labels, total_loss, total_accurate):
    cross_entropy = nn.BCELoss(reduction='sum')
    loss = cross_entropy(preds, labels)

    total_loss = total_loss + loss.detach().cpu().numpy()

    predicted_classes = (preds.detach().numpy() >= 0.5)

    accurate = sum(predicted_classes == np.array(labels).astype(bool))

    total_accurate = total_accurate + accurate

    return loss, total_accurate, total_loss
