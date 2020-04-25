from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import datetime
import json
import copy
import math
import random
import psutil
import time

import numpy as np
import pandas as pd

from keras.models import Sequential, model_from_json
#from keras.layers.core import TimeDistributedDense, Activation, Dense, Flatten, Merge
from keras.layers.core import Activation, Dense, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.constraints import maxnorm
from keras.regularizers import l2 #, activity_l2

from sklearn.model_selection import train_test_split
from sklearn import manifold
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data.dataloader import DataLoader

import wandb
from wandb.keras import WandbCallback
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from atacWorksModel import AtacWorksModel
from LSTMModel import LSTMModel
from EncoderDecoder import EncoderDecoder
from CnnEncoderDecoder import CnnEncoderDecoder
from AdvCnnEncoderDecoder import AdvCnnEncoderDecoder
from dataWithLabelsDataset import DataWithLabelsDataset
from kerasFormatConverter import KerasFormatConverter
from prepData import generate_bigWig, get_peaks, perform_denormalization, input_not_before_end
from dataset import DatasetEncoder
import evaluations
from dataNormalizer import DataNormalizer

from diConstants import (BASE_ROOT, MODELS_ROOT, WEIGHTS_ROOT,
                         RESULTS_ROOT, LOSS_ROOT, HIST_ROOT, EVAL_ROOT, RESULTS_BIGWIG_ROOT,
                         BIN_SIZE, GENOME_BATCH_SIZE, NUM_BASES, DEVICE)



def pad_sequence_with_zeros(X, padding):
    """
    Takes in a matrix X of shape num_bins x num_histone_marks and adds zero padding to the left end
    and to the right end. Returns a matrix of shape (num_bins + 2 * padding) x num_histone_marks 
    """

    assert len(X.shape) == 2
    assert padding >= 0

    num_bins, num_histone_marks = X.shape
    P = np.zeros([
        num_bins + 2 * padding,
        num_histone_marks])

    # Say we want to add a padding of 2 on each side of an X that is 101 x 6
    # We want P[2:103, :] to be X
    P[padding : (num_bins+padding), :] = X

    return P



class SeqModel(object):
    """
    Base class from which SeqToPoint derives. (We used this base class to prototype other 
    approaches; the paper is based only on SeqToPoint.)
    
    SeqToPoint:
        X is 3D with shape num_examples x seq_length x num_input_marks
        Y is 2D with shape num_examples x 1 x num_output_marks
    
    SeqModel implements instance methods:
        load_model()
        save_model_params()
        get_unprocessed_data()
        get_processed_data()
        train_single_model()
        compile_and_train_model()
        evaluate_model()
        test_model_on_samples()
        test_model_on_genome()

    Static method:
        instantiate_model()
    This is a static method because the __init__ function expects model_params,
    and if we're loading a model from a file, we don't know those model_params before we
    call this method.

    Abstract methods:
        process_X()
        process_Y()   
        predict_samples()
        predict_sequence()
    """

    def __init__(self, model_params):

        # We set the random seed two times in this file:
        # Once here, and once right after loading the training data but before training the model.
        # The reason is that when we try to load the training data, if the dataset doesn't exist yet,
        # we generate and save the training data on the fly. However, the dataset generation code
        # also sets the numpy random seed, so we need to reset it after loading the data.
        # We set a random seed of model_params['random_seed'] here 
        # and a random seed of model_params['random_seed'] + 42 right after loading the data.
        os.environ['PYTHONHASHSEED'] = str(model_params['random_seed'])
        random.seed(model_params['random_seed'])
        np.random.seed(model_params['random_seed'])
        torch.manual_seed(model_params['random_seed'])

        self.model_library = model_params['model_library']
        # if not (self.model_library in ['keras']):
        #     raise ValueError("model_library must be 'keras'")

        self.model = None
        self.model_params = model_params
        self.dataset_params = model_params['dataset_params']
        self.train_dataset = model_params['dataset_params']['train_dataset']
        self.test_datasets = model_params['dataset_params']['test_datasets']


        self.normalizer = DataNormalizer(self.model_params['scale_input'])

        # self.model_stamp is the unique identifier for this particular model
        # It looks like "RNN-20150911-175345976535", where the numbers are the date and time
        # that the model was saved, down to the microsecond to avoid race conditions.
        # It is set when the model is saved, and can be read from the filename that
        # the model is saved in.
        self.model_stamp = None

        # self.model_path is where the model was saved in the disk
        # It should be in MODELS_ROOT with filename [model_stamp],json
        self.model_path = None

        self.final_train_error = None
        self.final_valid_error = None

        self.hist = None

        self.input_marks = model_params['input_marks']
        self.num_input_marks = len(self.input_marks)

        self.output_marks = model_params['output_marks']
        self.num_output_marks = len(self.output_marks)

        assert(input_not_before_end(model_params['output_marks']))
        assert(input_not_before_end(model_params['input_marks']))

        self.is_output_in_input = True


        for output_mark in self.output_marks:
            if output_mark not in self.input_marks:
                self.is_output_in_input = False
                break

        if (self.model_params['predict_binary_output']) and ('INPUT' in self.output_marks):
            raise ValueError("Cannot predict peaks on INPUT.")

        self.verbose = True

        # print("Initialized model with parameters:")
        # print(json.dumps(model_params, indent=4, cls=DatasetEncoder))


    @staticmethod
    def instantiate_model(model_params):
        """
        Given model_params, looks at the model_class in it and 
        returns a copy of the appropriate subclass of SeqModel.
        """

        if model_params['model_class'] == 'SeqToSeq':
            m = SeqToSeq(model_params)
        elif model_params['model_class'] == 'SeqToPoint':
            m = SeqToPoint(model_params)

        return m


    def load_model(self, model_path):
        """
        Loads a Keras model from disk. 

        This only works on Keras models.

        The model will need to be compiled before it can be used for training.

        This is currently a weird function: because it's an instance method,
        it expects a SeqModel object to already exist. Worse, the SeqModel object
        must already be pre-initialized with fake model_params, since the SeqModel constructor
        needs model_params to be passed in. 

        This should be rewritten when we actually need to use it. 
        Thankfully, it is not super useful right now - we will only need it
        if the model init code changes such that we cannot recover previous models with 
        current code plus model_params.
        """

        assert self.model_library == 'keras'

        model_JSON_str = open(model_path).read()
        model_JSON = json.loads(model_JSON_str)

        self.model = model_from_json(model_JSON_str)
        self.model_params = model_JSON['_modelParams']
        self.dataset_params = self.model_params['dataset_params']
        assert self.model_params['model_library'] == 'keras'

        self.num_input_marks = self.model_params['num_input_marks']
        self.num_output_marks = self.model_params['num_output_marks']

        self.final_train_error = None
        self.final_valid_error = None

        self.hist = None

        # self.model_stamp is the unique identifier for this particular model
        # It looks like "RNN-20150911-175345976535", where the numbers are the date and time
        # that the model was saved, down to the microsecond to avoid race conditions.
        self.model_stamp = os.path.splitext(
            os.path.basename(model_path))[0]

        self.model_path = model_path

        return None


    def get_unprocessed_data(self, dataset):
        """
        Loads the train or test dataset (as specified in train_or_test) found in self.dataset_params
        in its original seq-to-seq form, as returned by extractDataset.load_seq_dataset.

        This function resets the random seed.
        """
        X, Y, peakPValueX, peakPValueY, peakBinaryX, peakBinaryY = dataset.load_seq_dataset(
            seq_length=self.dataset_params['seq_length'],
            input_marks=self.input_marks,
            output_marks=self.output_marks)

        if self.model_params['zero_out_non_bins']:
            peakPValueX = peakPValueX * peakBinaryX
            peakPValueY = peakPValueY * peakBinaryY

        if ((self.num_input_marks != X.shape[2]) or
            (self.num_input_marks != peakPValueX.shape[2] + ('INPUT' in self.input_marks))):
            raise Exception("num_input_marks between model and data needs to agree")
        if ((self.num_output_marks != Y.shape[2]) or
            (self.num_output_marks != peakPValueY.shape[2] + ('INPUT' in self.output_marks))):
            raise Exception("num_output_marks between model and data needs to agree")

        # See comment in __init__ about random seeds
        os.environ['PYTHONHASHSEED'] = str(self.model_params['random_seed'] + 42)
        random.seed(self.model_params['random_seed'] + 42)
        np.random.seed(self.model_params['random_seed'] + 42)
        torch.manual_seed(self.model_params['random_seed'] + 42)

        return (X, Y, peakPValueX, peakPValueY, peakBinaryX, peakBinaryY)


    def get_processed_data(self, dataset):
        """
        Returns the train or test dataset (as specified in train_or_test) found in 
        self.dataset_params, transformed into a format that the model can directly use.

        Helper functions process_X and process_Y are implemented in subclasses because
        different models need differently formatted data, e.g., seq-to-seq vs. seq-to-point.
        
        Seq-to-seq takes in:
            X: num_examples x seq_length x num_input_marks
            Y: num_examples x seq_length x num_output_marks

        Seq-to-point takes in:
            X: num_examples x seq_length x num_input_marks
            Y: num_examples x 1 x num_output_marks

        Point-to-point takes in:
            X: num_examples x (seq_length * num_input_marks)
            Y: num_examples x num_output_marks

        """

        X, Y, peakPValueX, peakPValueY, peakBinaryX, peakBinaryY = self.get_unprocessed_data(dataset)

        X = self.process_X(X)
        Y = self.process_Y(Y)
        peakPValueX = self.process_X(peakPValueX)
        peakPValueY = self.process_Y(peakPValueY)
        peakBinaryX = self.process_X(peakBinaryX)
        peakBinaryY = self.process_Y(peakBinaryY)

        return (X, Y, peakPValueX, peakPValueY, peakBinaryX, peakBinaryY)


    def compile_and_train_model(self):
        """
        Trains the model specified by self.model and self.model_params 
        on the training data given by self.dataset_params.

        If self.model is a Keras model, it also writes out model weights and 
        training history to disk.
        """

        assert self.model
        assert self.model_params

        # Train model
        (train_X, Y, peakPValueX, peakPValueY, peakBinaryX, peakBinaryY) = self.get_processed_data(
            self.train_dataset)

        if self.model_params['model_type'] != 'adv-cnn-encoder-decoder':
            self.normalizer.fit(train_X)
            train_X = self.normalizer.transform(train_X)
        train_inputs_X = train_X

        if self.model_params['predict_binary_output']:
            train_Y = peakBinaryY
        else:
            train_Y = Y

        if self.model_params['train_params']['nb_epoch'] == 0:
            return None

        if self.model_library == 'keras':

            # Compiles model: this sets the optimizer and loss function
            self.model.compile(**self.model_params['compile_params'])

            # ModelCheckpoint() is a Keras callback that saves the weights of the model while 
            # it's being trained.
            # save_best_only means that the model weights will be saved after every epoch
            # in which the validation error improves.
            checkpointer = ModelCheckpoint(
                filepath=os.path.join(WEIGHTS_ROOT, '%s-weights.tf' % self.model_stamp),
                verbose=1,
                save_best_only=True)

            # EarlyStopping() is a Keras callback that stops training once the validation loss
            # of the model has not improved for [patience] epochs in a row.
            earlystopper = EarlyStopping(monitor='val_loss', patience=3, verbose=0)

            if self.model_params['train_params']['wandb_log']:
                wandbcallback = WandbCallback()
                callbacks = [checkpointer, earlystopper, wandbcallback]
            else:
                callbacks = [checkpointer, earlystopper]

            self.hist = self.model.fit(
                train_inputs_X,
                train_Y,
                callbacks=callbacks,
                nb_epoch=self.model_params['train_params']['nb_epoch'],
                batch_size=self.model_params['train_params']['batch_size'],
                validation_split=self.model_params['train_params']['validation_split'])

            # Store training history for Keras models
            # Note that the "final training error" in self.hist.history is only approximate: 
            # it is averaged over all minibatches in the final epoch. So it's not exactly the 
            # training error with the final weights. The final validation error is accurate.

            hist_path = os.path.join(
                HIST_ROOT,
                "%s.hist" % self.model_stamp)

            with open(hist_path, 'w') as f:
                f.write(json.dumps(self.hist.history))

        elif self.model_library == 'pytorch':
            self.model = self.model.to(DEVICE)

            lr = self.model_params['compile_params']['lr']
            
            if self.model_params['compile_params']['optimizer'] == 'adagrad':
                optimizer = optim.Adagrad(self.model.parameters(), lr=lr)
            elif self.model_params['compile_params']['optimizer'] == 'adam':
                optimizer = optim.Adam(self.model.parameters(), lr=lr)
            if self.model_params['compile_params']['loss'] == 'binary_crossentropy':
                loss_function = torch.nn.modules.loss.BCELoss()
            elif self.model_params['compile_params']['loss'] == 'MSE':
                loss_function = torch.nn.modules.loss.MSELoss()

            nb_epoch = self.model_params['train_params']['nb_epoch']
            batch_size = self.model_params['train_params']['batch_size']

            self.hist = self.fit_model(train_inputs_X, train_Y,
                                       optimizer, loss_function,
                                       nb_epoch, batch_size)

            hist_path = os.path.join(
                HIST_ROOT,
                "%s.hist" % self.model_stamp)

            with open(hist_path, 'w') as f:
                f.write(json.dumps(self.hist))

        return None

    def fit_model(self, train_inputs_X, train_Y, optimizer, loss_function, nb_epoch, batch_size):
        assert self.model_library == 'pytorch'

        train_data, val_data, train_labels, val_labels = train_test_split(
            train_inputs_X, train_Y, test_size=self.model_params['train_params']['validation_split'])

        train_dataset = DataWithLabelsDataset(train_data, train_labels)
        val_dataset = DataWithLabelsDataset(val_data, val_labels)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # pid = os.getpid()
        # py = psutil.Process(pid)
        # memoryUse = py.memory_info()[0] / 2.0 ** 30
        # print("before train loop ", memoryUse)

        checkpoint_path = os.path.join(WEIGHTS_ROOT, '%s-weights.pt' % self.model_stamp)
        earlystopper_patience = 3
        best_epoch_val_loss = None
        hist = {'loss': [], 'val_loss': []}
        params_num = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model parameters: {params_num}")
        print(f"Model checkpoint path: {checkpoint_path}")

        if self.model_params['model_type'] == 'adv-cnn-encoder-decoder':
            lr = self.model_params['compile_params']['lr']
            gen_optimizer = optim.Adam(self.model.parameters(), lr=lr * 2)
            disc_optimizer = optim.Adam(self.model.discriminator.parameters(), lr=lr / 2)
            disc_loss_function = torch.nn.modules.loss.BCELoss()
            output_marks_idx = [self.input_marks.index(output_mark) for output_mark in self.output_marks]

        for epoch in tqdm(range(nb_epoch)):
            loss_values = []
            val_loss_values = []

            self.model.train()
            if self.model_params['model_type'] == 'adv-cnn-encoder-decoder':
                disc_fool_loss_values = []
                noisy_latent_vectors = []
                clean_latent_vectors = []
                batch_i = 0
                
                for batch_data in tqdm(train_loader):
                    batch_i += 1
                    optimizer.zero_grad()
                    data, labels = batch_data[0].to(DEVICE), batch_data[1].to(DEVICE)
                    output, latent_noisy = self.model(data, return_latent=True)
                    if (batch_i % 1000) == 0:
                        loss = loss_function(output, labels.float())
                        loss.backward(retain_graph=True)
                        optimizer.step()
                        loss_values.append(loss.item())
                    
                    gen_optimizer.zero_grad()
                    batch_data_size = data.shape[0]
                    gen_noisy_output = self.model.discriminator(latent_noisy)
                    gen_noisy_loss = disc_loss_function(gen_noisy_output, torch.ones((batch_data_size, 1)).to(DEVICE))
                    gen_noisy_loss.backward()
                    gen_optimizer.step()
                    disc_fool_loss_values.append(gen_noisy_loss.item())

                    #clean_data = copy.deepcopy(data)
                    #print(clean_data.gather(2, torch.tensor(output_marks_idx * (data.shape[0] * data.shape[1])).view(data.shape[0], data.shape[1], 1).to(DEVICE)))
                    #clean_data[:, :, torch.tensor(output_marks_idx).to(DEVICE)] = copy.deepcopy(labels).float()
                    # TODO normalize
                    #_, latent_clean = self.model(clean_data, return_latent=True)
                    latent_clean = torch.from_numpy(np.random.normal(0.0, 1.0, latent_noisy.shape)).float().to(DEVICE)
                    
                    disc_optimizer.zero_grad()
                    disc_clean_output = self.model.discriminator(latent_clean.detach())
                    disc_clean_loss = disc_loss_function(disc_clean_output, torch.ones((batch_data_size, 1)).to(DEVICE))
                    disc_clean_loss.backward(retain_graph=True)
                    disc_noisy_output = self.model.discriminator(latent_noisy.detach())
                    disc_noisy_loss = disc_loss_function(disc_noisy_output, torch.zeros((batch_data_size, 1)).to(DEVICE))
                    disc_noisy_loss.backward()
                    disc_optimizer.step()
                    
                    if len(noisy_latent_vectors) < 20:
                        noisy_latent_vectors.append(latent_noisy.detach().view(batch_data_size, -1))
                        clean_latent_vectors.append(latent_clean.detach().view(batch_data_size, -1))
                    #else:
                    #    print("breaking")
                    #    break
                
                print(f"Epoch: {epoch} Disc fool loss: {np.mean(disc_fool_loss_values)}")
                if epoch == 0 or epoch == nb_epoch - 1 or (epoch % 5) == 0:
                    noisy_latent_vectors_cat = torch.cat(noisy_latent_vectors, dim=0)
                    clean_latent_vectors_cat = torch.cat(clean_latent_vectors, dim=0)
                    all_latent = torch.cat((noisy_latent_vectors_cat, clean_latent_vectors_cat), dim=0)
                    print(f"All latent shape: {all_latent.shape}")
                    
                    manifold_method = manifold.Isomap(n_neighbors=10, n_components=2)
                    t0 = time.time()
                    latent_transformed = manifold_method.fit_transform(all_latent.cpu().numpy())
                    t1 = time.time()
                    print(f"Manifold method time: {t1 - t0}sec Num vectors: {latent_transformed.shape[0]}")
                    plt.scatter(latent_transformed[latent_transformed.shape[0] // 2:, 0], 
                                latent_transformed[latent_transformed.shape[0] // 2:, 1], s=10, c='black')
                    plt.scatter(latent_transformed[:latent_transformed.shape[0] // 2, 0], 
                                latent_transformed[:latent_transformed.shape[0] // 2, 1], s=10, c='red')
                    #plt.show()
                    if self.model_params['train_params']['wandb_log']:
                        wandb.log({f'Latent at #{epoch}': wandb.Image(plt)}, step=epoch)
                    plt.clf()
                    
            else:
                for batch_data in tqdm(train_loader):
                    optimizer.zero_grad()
                    data, labels = batch_data[0].to(DEVICE), batch_data[1].to(DEVICE)

                    if self.model_params['model_type'] == 'encoder-decoder':
                        output = self.model(data, labels)
                    else:
                        output = self.model(data)

                    loss = loss_function(output, labels.float())

                    # pid = os.getpid()
                    # py = psutil.Process(pid)
                    # memoryUse = py.memory_info()[0] / 2.0 ** 30
                    # print("before backward ", memoryUse)
                    loss.backward()
                    # pid = os.getpid()
                    # py = psutil.Process(pid)
                    # memoryUse = py.memory_info()[0] / 2.0 ** 30
                    # print("before step ", memoryUse)
                    optimizer.step()
                    
                    if torch.sum(output[output != 0.0]) == 0.0:
                        print("sum(output) == 0, expected: ", torch.sum(labels[labels != 0.0]))
                    #for param in self.model.parameters():
                    #    print(param.data)
                    #    break
                    # pid = os.getpid()
                    # py = psutil.Process(pid)
                    # memoryUse = py.memory_info()[0] / 2.0 ** 30
                    # print("after step", memoryUse)
                    loss_values.append(loss.item())
                
            epoch_train_loss = np.mean(loss_values)
            hist['loss'].append(float(epoch_train_loss))

            self.model.eval()
            with torch.no_grad():
                for batch_data in tqdm(val_loader):
                    data, labels = batch_data[0].to(DEVICE), batch_data[1].to(DEVICE)

                    output = self.model(data)

                    loss = loss_function(output, labels.float())
                    val_loss_values.append(loss.item())

                epoch_val_loss = np.mean(val_loss_values)
                hist['val_loss'].append(float(epoch_val_loss))
                if self.model_params['train_params']['wandb_log']:
                    wandb.log({'loss': epoch_train_loss}, step=epoch)
                    wandb.log({'val_loss': epoch_val_loss}, step=epoch)
                    wandb.log({'epoch': epoch}, step=epoch)
                print(f"Epoch: {epoch} Loss: {epoch_train_loss} Val loss: {epoch_val_loss}")

                if best_epoch_val_loss is None or best_epoch_val_loss > epoch_val_loss:
                    best_epoch_val_loss = epoch_val_loss
                    torch.save(self.model.state_dict(), checkpoint_path)

                if epoch == 1 and abs(hist['val_loss'][1] - hist['val_loss'][0]) <= 1e-9:
                    print(f"Did not improve from {hist['val_loss'][0]} to {hist['val_loss'][1]}, breaking")
                    #break
                # if earlystopper_patience == 0:
                #     break
        
        print(f"Model checkpoint path: {checkpoint_path}")
        self.model.load_state_dict(torch.load(checkpoint_path))
        self.model.eval()
        return hist

    def save_model_params(self):
        """
        Writes model to disk, initializing model_stamp and model_path in the process.
        This function is called in the __init__ method of derived classes.

        For Keras models, this saves compilation parameters separately 
        without actually compiling the model to save time. Keras model weights are
        saved during training through the ModelCheckpoint() callback, so we can reconstruct
        trained models by separately loading the saved params and the weights. See
        http://keras.io/faq/#how-can-i-save-a-keras-model for more details.
        """

        assert self.model
        assert self.model_params

        timeStr = datetime.datetime.now().strftime("%Y%m%d-%H%M%S%f")
        self.model_stamp = "%s-%s" % (self.model_params['model_type'], timeStr)

        # If it's a Keras model, we save not only model_params but the actual
        # architecture of the model, since the code that constructs models from model_params
        # might change over time.
        if self.model_library == 'keras':
            model_JSON = self.model_params
            model_JSON['_keras_model_params'] = json.loads(self.model.to_json())
            model_JSON_str = json.dumps(model_JSON, cls=DatasetEncoder)

            self.model_path = os.path.join(MODELS_ROOT, "%s.json" % self.model_stamp)

            assert os.path.isfile(self.model_path) == False

            with open(self.model_path, 'w') as model_file:
                model_file.write(model_JSON_str)
        elif self.model_library == 'pytorch':
            if self.model_params['pretrained_model_path'] is not None:
                self.model.load_state_dict(torch.load(self.model_params['pretrained_model_path']))

            self.model_path = os.path.join(MODELS_ROOT, "%s.pt" % self.model_stamp)

            assert os.path.isfile(self.model_path) == False

            # torch.save(self.model, self.model_path)

        return None


    def test_model_on_samples(self, dataset, train_or_test):
        """
        Evaluates the model on samples drawn from dataset.
        Returns a dictionary with keys orig_results and denoised_results, with values obtained
        from evaluations.compare.

        The train_or_test param is just for display.
        """
        assert self.model
        assert train_or_test == 'train' or train_or_test == 'test'

        (X, Y, peakPValueX, peakPValueY, peakBinaryX, peakBinaryY) = self.get_unprocessed_data(dataset)
        binaryY = peakBinaryY

        Y = self.process_Y(Y)
        peakPValueY = self.process_Y(peakPValueY)
        binaryY = self.process_Y(binaryY)

        if not self.model_params['predict_binary_output']:
            print('Bias-only MSE is ', np.mean((Y - np.mean(Y)) ** 2))

        # First, compare the true data with the subsampled data
        # To get the "original" error, we just make the prediction that Y = X.
        # Before doing this, we have to call process_Y on X to get it into the right form.
        # This is not a typo! We have to process X in the way that we'd normally process Y.
        # This is needed for SeqToPoint and PointToPoint models, since in those models 
        # the X and Y returned from self.get_data have different shapes.
        # Since input_marks might not equals output_marks, we also have to subset the right
        # parts of X to compare.
        # If we're doing de novo imputation then output marks will not be in input marks; if
        # so, we just skip this step.
        orig_results = None

        if self.is_output_in_input:
            output_marks_idx = [self.input_marks.index(output_mark) for output_mark in self.output_marks]
            if self.model_params['predict_binary_output']:
                print("%s samples - Original peaks vs. true peaks:" % train_or_test)
                orig_results = evaluations.compare(
                    self.process_Y(peakPValueX[..., output_marks_idx]),
                    binaryY,
                    predict_binary_output=True)
            else:
                print("%s samples - Original:" % train_or_test)
                orig_results = evaluations.compare(
                    self.process_Y(X[..., output_marks_idx]),
                    Y,
                    predict_binary_output=False)

        # Then compare the true data with the output of the model
        # Process the data properly
        X = self.process_X(X)
        if self.model_params['model_type'] != 'adv-cnn-encoder-decoder':
            X = self.normalizer.transform(X)

        # We have to batch the prediction so that the GPU doesn't run out of memory
        if 'eval_batch_size' in self.model_params['train_params']:
            batch_size = self.model_params['train_params']['eval_batch_size']
        else:
            batch_size = 10000
        num_examples = X.shape[0]
        num_batches = int(math.ceil(1.0 * num_examples / batch_size))

        # If predict_binary_output is true, then INPUT cannot be in output_marks, so
        # Y will have the same shape as binaryY. 
        # This is not necessarily true if predict_binary_output is false.
        # There's no need to branch separately here to initialize Y_pred = np.empty(binaryY.shape) 
        # if predict_binary_output is true.
        Y_pred = np.empty(Y.shape)

        for batch in tqdm(range(num_batches)):
            start_idx = batch * batch_size
            end_idx = min((batch + 1) * batch_size, num_examples)
            Y_pred[start_idx : end_idx] = self.predict_samples(X[start_idx : end_idx])
            # Y_pred[start_idx : end_idx] = self.predict_samples(X[start_idx : end_idx], torch.device('cpu'))

        if self.model_params['predict_binary_output']:
            print("%s samples - Predicted peaks vs. true peaks:" % train_or_test)
            denoised_results = evaluations.compare(Y_pred, binaryY, predict_binary_output=True)
        else:
            print("%s samples - Denoised:" % train_or_test)
            denoised_results = evaluations.compare(Y_pred, Y, predict_binary_output=False)

        samples_results = {
            'orig': orig_results,
            'dn': denoised_results
        }
        return samples_results


    def test_model_on_genome(self, dataset):
        """
        Evaluates the model on the entire genome in dataset.
        Returns a dictionary with keys orig_results and denoised_results, with values obtained
        from evaluations.compare.

        This function generates genome-wide predictions for each chromosome in the 
        test dataset. Blacklisted regions have previously been zero-ed out in prepData.
        """

        assert self.model

        # only_chr1 controls whether genome-wide prediction is done on the whole genome, or just
        # on chr1 for speed.
        only_chr1 = self.dataset_params['only_chr1']
        # Load data   
        test_X_all = dataset.load_genome(
            "X",
            marks=self.input_marks,
            only_chr1=only_chr1,
            peaks=False)

        if self.model_params['predict_binary_output']:
            #if binary, want to use binary peak matrix as Y.
            #and noisy peak p-values as baseline.
            assert('INPUT' not in self.output_marks)
            test_Y_all, _ = dataset.load_binary_genome(
                "Y",
                marks=self.output_marks,
                only_chr1=only_chr1)

            noisy_peak_pvals_all = dataset.load_genome(
                "X",
                marks=self.output_marks,
                only_chr1=only_chr1,
                peaks=True)

            if self.model_params['zero_out_non_bins']:
                noisy_peaks_all, _ = dataset.load_binary_genome(
                    "X",
                    marks=self.output_marks,
                    only_chr1=only_chr1)
                assert(set(noisy_peak_pvals_all.keys()) == set(noisy_peaks_all.keys()))
                for chrom in noisy_peak_pvals_all:
                    noisy_peak_pvals_all[chrom] = noisy_peak_pvals_all[chrom] * noisy_peaks_all[chrom]

        else:
            #otherwise, use continuous non-subsampled signal as Y.
            test_Y_all = dataset.load_genome(
                "Y",
                marks=self.output_marks,
                only_chr1=only_chr1,
                peaks=False)

            # Load peaks from test cell line
            peak_locs_all = {}
            for factor in dataset.marks_in_dataset:
                if factor == 'INPUT': continue
                peak_locs, _ = get_peaks(
                    dataset.cell_line,
                    factor,
                    subsample_target_string=dataset.Y_subsample_target_string)
                peak_locs_all[factor] = peak_locs


        if (test_Y_all.keys() != test_X_all.keys()):
            raise Exception("Subsampled and full data must have the same chroms")


        chroms = sorted(test_X_all.keys())


        ### Compute results separately for each chromosome

        orig_results_all = {}
        denoised_results_all = {}
        orig_results_peaks = {}
        denoised_results_peaks = {}

        preds = {}

        # Warning: the peak comparison code relies on the sequence starting at the start
        # of the chromosome. If this is not true, we'd have to offset the peak coordinates before
        # passing them into compare().
        if only_chr1:
            chroms = ['chr1']

        for chrom in chroms:
            test_X = test_X_all[chrom]
            test_Y = test_Y_all[chrom]
            if self.model_params['predict_binary_output']:
                noisy_peak_pvals = noisy_peak_pvals_all[chrom]


            if self.dataset_params['num_bins_to_test']:
                num_bins_to_test = self.dataset_params['num_bins_to_test']
                assert num_bins_to_test > 0
                test_X = test_X[:num_bins_to_test]
                test_Y = test_Y[:num_bins_to_test]
                if self.model_params['predict_binary_output']:
                    noisy_peak_pvals = noisy_peak_pvals[:num_bins_to_test]

            assert test_X.shape[0] == test_Y.shape[0], \
                "Subsampled and full data must have the same length"

            if self.model_params['predict_binary_output']:
                assert(list(noisy_peak_pvals.shape) == list(test_Y.shape))

            assert test_X.shape[1] == self.num_input_marks
            assert test_Y.shape[1] == self.num_output_marks

            chrom_length = test_X.shape[0]

            ### Get a list of peaks for this chromosome            
            peaks = []
            if not self.model_params['predict_binary_output']:
                for factor in self.output_marks:
                # For INPUT, we calculate MSE across peaks of all other marks in the test dataset,
                # since we want to get INPUT right whenever there's a peak in some other mark.
                # Note that we're purely concatenating peaks from different marks here,
                # so there'll be some overlapping peaks.
                # This is fine right now but might break later depending on what evaluation code we
                # write, so watch out.
                    if factor == 'INPUT':
                        peak_factor = []
                        for other_factor in dataset.marks_in_dataset:
                            if other_factor == 'INPUT': continue
                            peak_factor.extend(peak_locs_all[other_factor][chrom])
                        peak_factor = np.array(peak_factor)
                    else:
                        peak_factor = peak_locs_all[factor][chrom]
                    peaks.append(peak_factor)

            ### Do comparisons between original (subsampled) and full data
            # The original comparison is only done if the output mark is actually in the input data
            if self.is_output_in_input:
                if not self.model_params['predict_binary_output']:

                    output_marks_idx = [self.input_marks.index(output_mark) for output_mark in self.output_marks]

                    print("Test %s, %.2E bins - Original, all signal:" % (chrom, chrom_length))
                    orig_results_all[chrom] = evaluations.compare(
                        test_X[:, output_marks_idx],
                        test_Y,
                        predict_binary_output=False)

                    print("Test %s, %.2E bins - Original, only peaks:" % (chrom, chrom_length))
                    orig_results_peaks[chrom] = evaluations.compare(
                        test_X[:, output_marks_idx],
                        test_Y,
                        predict_binary_output=False,
                        peaks=peaks)

                elif self.model_params['predict_binary_output']:
                    print("Test %s, %.2E bins - Original:" % (chrom, chrom_length))
                    orig_results_all[chrom] = evaluations.compare(
                        noisy_peak_pvals,
                        test_Y,
                        predict_binary_output=True)


            ### Do comparisons between model output and full data
            # We have to batch this up so that the GPU doesn't run out of memory
            # Assume a fixed batch size of 5M bins
            num_batches = int(math.ceil(1.0 * chrom_length / GENOME_BATCH_SIZE))

            test_Y_pred = np.empty(test_Y.shape)
            if self.model_params['model_type'] != 'adv-cnn-encoder-decoder':
                test_X = self.normalizer.transform(test_X)

            print(num_batches, GENOME_BATCH_SIZE)

            for batch in tqdm(range(num_batches)):
                start_idx = batch * GENOME_BATCH_SIZE
                end_idx = min((batch + 1) * GENOME_BATCH_SIZE, chrom_length)
                test_Y_pred[start_idx : end_idx] = self.predict_sequence(
                    test_X[start_idx : end_idx])
                # test_Y_pred[start_idx : end_idx] = self.predict_sequence(
                #     test_X[start_idx : end_idx], torch.device('cpu'))
                # with np.printoptions(precision=3):
                #     print('###########')
                #     #print(test_X[end_idx - 50 : end_idx])
                #     print(test_Y[end_idx - 50 : end_idx].flatten())
                #     print(test_Y_pred[end_idx - 50 : end_idx].flatten())
                
                
            print("Test %s, %.2E bins - Denoised, all signal:" % (chrom, chrom_length))
            denoised_results_all[chrom] = evaluations.compare(
                test_Y_pred,
                test_Y,
                predict_binary_output=self.model_params['predict_binary_output'])

            if not self.model_params['predict_binary_output']:
                print("Test %s, %.2E bins - Denoised, only peaks:" % (chrom, chrom_length))
                denoised_results_peaks[chrom] = evaluations.compare(
                    test_Y_pred,
                    test_Y,
                    predict_binary_output=False,
                    peaks=peaks)



            # If we're generating a bigWig file from the output, we need to save the results
            # If we're doing regression, we first denormalize the outputs so that it can be viewed v
            # correctly in the genome browser
            if self.model_params['generate_bigWig']:

                if self.model_params['predict_binary_output']:
                    preds[chrom] = test_Y_pred
                else:
                    preds[chrom] = perform_denormalization(
                        test_Y_pred,
                        dataset.normalization)

        # Write bigWig file to disk
        if self.model_params['generate_bigWig']:
            if self.model_params['predict_binary_output']:
                suffix = 'peaks'
            else:
                suffix = 'signal'

            generate_bigWig(
                preds,
                self.output_marks,
                '%s_%s_subsample-%s_%s' % (
                    self.model_stamp,
                    dataset.cell_line,
                    dataset.X_subsample_target_string ,
                    suffix),
                RESULTS_BIGWIG_ROOT)

        # Construct dict of results
        if self.model_params['predict_binary_output']:
            test_genome_results = {
                'orig_all': orig_results_all,
                'dn_all': denoised_results_all
            }
        else:
            test_genome_results = {
                'orig_all': orig_results_all,
                'dn_all': denoised_results_all,
                'orig_peaks': orig_results_peaks,
                'dn_peaks': denoised_results_peaks,
            }



        print('final results', test_genome_results)
        return test_genome_results


    def evaluate_model(self, evaluate_genome_only):
        """
        Evaluates the model on the train and test datasets specified in self.dataset_params.
        Writes the results to disk in EVAL_ROOT.        
        """
        # We need to write our own JSON encoder for numpy.float32s
        # because the built-in JSON encoder only knows how to encode normal floats
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.floating):
                    return float(obj)
                else:
                    return super(NumpyEncoder, self).default(obj)

        if not evaluate_genome_only:
            # Evaluate model on training data        
            train_samples_results = self.test_model_on_samples(self.train_dataset, 'train')
            train_results = {
                'samples': train_samples_results
            }

            train_eval_path = os.path.join(
                EVAL_ROOT,
                "%s-train.eval" % self.model_stamp)

            with open(train_eval_path, 'w') as f:
                f.write(json.dumps(train_results, cls=NumpyEncoder))

        # Evaluate model on testing data
        all_test_results = []
        for dataset_idx, test_dataset in enumerate(self.test_datasets):
            if not evaluate_genome_only:
                test_samples_results = self.test_model_on_samples(test_dataset, 'test')

            try:
                test_genome_results = self.test_model_on_genome(test_dataset)
            except NotImplementedError:
                print("Genome-wide prediction hasn't been implemented for this type of model. Skipping...")
                test_genome_results = None

            if not evaluate_genome_only:
                test_results = {
                    'samples': test_samples_results,
                    'genome': test_genome_results
                }
            else:
                test_results = {
                    'genome': test_genome_results
                }
                
            test_eval_path = os.path.join(
                EVAL_ROOT,
                "%s-test-%s.eval" % (self.model_stamp, dataset_idx))
            with open(test_eval_path, 'w') as f:
                f.write(json.dumps(test_results, cls=NumpyEncoder))

            all_test_results.append(test_results)

        if not evaluate_genome_only:
            results = {
                'train_samples': train_samples_results,
                'test_results': all_test_results
            }
        else:
            results = {
                'test_results': all_test_results
            }
            
        return results


    def process_X(self, X):
        """
        Takes in a matrix X of shape num_examples x seq_length x num_histone_marks,
        returned from extractDataset.load_seq_dataset, and processes it as necessary 
        for the type of model. X should be the input data that is fed to the model.
        
        This is implemented in subclasses because different models need differently
        formatted data, e.g., seq-to-seq vs. seq-to-point.
        """

        raise NotImplementedError


    def process_Y(self, Y):
        """
        Takes in a matrix Y of shape num_examples x seq_length x num_histone_marks,
        returned from extractDataset.load_seq_dataset, and processes it as necessary 
        for the type of model. Y should represent the desired output of the model.
        
        This is implemented in subclasses because different models need differently
        formatted data, e.g., seq-to-seq vs. seq-to-point.
        """

        raise NotImplementedError


    def SeqToX_predict_samples(self, signalX, device=None):
        """
        Common code used in the predict_samples() method defined in SeqToSeq and SeqToPoint subclasses.
        """

        num_examples = signalX.shape[0]

        assert len(signalX.shape) == 3
        assert signalX.shape[0] == num_examples
        assert signalX.shape[1] == self.dataset_params['seq_length']
        assert signalX.shape[2] == self.num_input_marks

        if self.model_library == 'keras':
            Y = self.model.predict(signalX)
        elif self.model_library == 'pytorch':
            if device is None:
                device = DEVICE
            self.model = self.model.to(device)
            self.model.eval()
            with torch.no_grad():
                X = torch.from_numpy(signalX).float().to(device)
                Y = self.model(X).detach().cpu().numpy()

        assert Y.shape[0] == num_examples
        assert Y.shape[2] == self.num_output_marks

        return Y


    def predict_samples(self, signalX, device=None):
        """
        Takes in input signalX of whatever dimensions are needed for the model, which
        is subclass-dependent. It passes it through the model and returns the output matrix.
        """

        raise NotImplementedError


    def predict_sequence(self, signalX, device=None):
        """
        Takes in input matrix signalX of dimensions num_bins x num_input_marks 
        and passes it through the model, 
        returning an output matrix of num_bins x num_output_marks.
        """

        raise NotImplementedError




class SeqToPoint(SeqModel):

    def __init__(self, model_params):
        """
        Initializes the correct model based on model_params.
        """

        super(SeqToPoint, self).__init__(model_params)

        assert self.dataset_params['seq_length'] % 2 == 1, "seq_length must be odd for SeqToPoint models."

        if model_params['model_type'] == 'cnn':

            num_filters = model_params['num_filters']
            filter_length = model_params['filter_length']

            model = Sequential()

            # border_mode='same' makes the length of the output 
            # the same size as the length of the input
            # by adding just the right amount of zero padding to each side.
            model.add(
                Convolution1D(
                    num_filters,
                    filter_length,
                    input_dim=self.num_input_marks,
                    init='uniform',
                    border_mode='same'))

            model.add(Activation('relu'))

            # See below for documentation on border_mode='valid'
            # We are essentially replicating the "dense" layer here, but with a convolutional layer
            # so that later we can do genome-wide prediction.
            model.add(
                Convolution1D(
                    self.num_output_marks, # output_dim,
                    self.dataset_params['seq_length'],
                    init='uniform',
                    border_mode='valid'))

            if model_params['predict_binary_output']:
                model.add(Activation('sigmoid'))
            else:
                model.add(Activation('relu'))

        # 'lrnn' stands for linear regression neural network
        # It is a single convolutional layer with filters that span the entire seq length. 
        # Essentially, this replicates linear or logistic regression in the Keras framework.
        # border_mode='valid' means that it only does convolutions where the whole filter can fit in the sequence 
        # so effectively it is only doing one convolution/feedforward operation during training. 
        # We make it convolutional so that we can easily do genome-wide predictions later.
        # It has as many neurons as there are histone marks, that is, there is one filter per histone mark.
        # This way, each histone mark gets seq_length * num_input_marks parameters to make a linear prediction.
        elif model_params['model_type'] == 'lrnn':
            model = Sequential()

            model.add(
                Convolution1D(
                    self.num_output_marks, # nb_filter: one filter per histone mark
                    self.dataset_params['seq_length'], # filter_length
                    input_dim=self.num_input_marks,
                    border_mode='valid'))

            if model_params['predict_binary_output']:
                model.add(Activation('sigmoid'))

        elif model_params['model_type'] == 'coda':
            num_filters = model_params['num_filters']
            filter_length = model_params['filter_length']
            seq_length = self.dataset_params['seq_length']

            if model_params['predict_binary_output']:
                last_activation = nn.Sigmoid()
            else:
                last_activation = nn.ReLU()

            model = nn.Sequential(
                KerasFormatConverter(),
                nn.Conv1d(
                    in_channels=self.num_input_marks,
                    out_channels=num_filters,
                    kernel_size=filter_length,
                    padding=filter_length // 2),
                nn.ReLU(),
                nn.Conv1d(
                    in_channels=num_filters,
                    out_channels=self.num_output_marks,
                    kernel_size=seq_length,
                    padding=0),
                last_activation,
                KerasFormatConverter())

        elif model_params['model_type'] == 'atac':
            num_filters = model_params['num_filters']
            filter_length = model_params['filter_length']
            seq_length = self.dataset_params['seq_length']
            predict_binary_output = model_params['predict_binary_output']

            model = AtacWorksModel(predict_binary_output=predict_binary_output,
                                   interval_size=seq_length,
                                   in_channels=self.num_input_marks,
                                   out_channels=self.num_output_marks,
                                   num_filters=num_filters,
                                   kernel_size=filter_length)

        elif model_params['model_type'] == 'cnn-encoder-decoder':
            predict_binary_output = model_params['predict_binary_output']
            hidden_size = model_params['hidden_size']
            kernel_size = model_params['kernel_size']
            stride = model_params['stride']
            dilation = model_params['dilation']
            num_layers = model_params['num_layers']
            residual = model_params['residual']
            dropout = model_params['dropout']
            seq_length = self.dataset_params['seq_length']

            model = CnnEncoderDecoder(
                predict_binary_output=predict_binary_output,
                in_channels=self.num_input_marks,
                out_channels=self.num_output_marks,
                hidden_size=hidden_size,
                num_layers=num_layers,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                residual=residual,
                p_dropout=dropout,
                seq_length=seq_length,
                seq2seq=False
            )

        else:
            raise Exception("Model type not recognized")

        self.model = model
        self.save_model_params()


    def process_X(self, X):
        """
        See documentation in SeqModel.
        Input to seq-to-point models need no further processing from load_seq_dataset.
        """

        return X


    def process_Y(self, Y):
        """
        See documentation in SeqModel.
        Takes in matrix Y of shape num_examples x seq_length x num_histone_marks
        and returns matrix of shape num_examples x 1 x num_histone_marks, selecting the 
        middle of the sequence.
        
        We want the singleton dimension so that we can avoid flattening the output of the 
        model in Keras. This doesn't matter in training, but it does in testing when we 
        are trying to do genome-wide predictions.
        """

        # If seq_length is 101
        # then the array goes from 0 to 100
        # and we want to pick mid = 50        
        mid = (self.dataset_params['seq_length'] - 1) // 2

        # Y = np.squeeze(Y[:, mid, :]) 
        # return Y

        return Y[:, mid:mid+1, :]


    def predict_samples(self, signalX, device=None):
        """
        Takes in input matrix signalX, of shape num_examples x seq_length x num_input_marks             
        and feeds it through the model, returning an output matrix of shape
        num_examples x 1 x num_output_marks.
        """
        Y = self.SeqToX_predict_samples(signalX, device=device)
        assert Y.shape[1] == 1

        return Y


    def predict_sequence(self, signalX, device=None):
        """
        Takes in input matrix signalX of dimensions num_bins x num_input_marks 
        and passes it through the model, 
        returning an output matrix of num_bins x num_output_marks.
        """
        # if ('lrnn' not in self.model_params['model_type']) and ('cnn' not in self.model_params['model_type']):
        #     raise NotImplementedError

        # We have to do some zero-padding on the input sequences before we pass them to the 
        # convolutional models defined in SeqToPoint. 
        # This is because the final layer of these conv nets are 'valid' convolutions with 
        # filter_length = seq_length. This means that the output of that layer, and therefore the 
        # model, will be (seq_length - 1) shorter than the input to that layer. This is necessary 
        # for training, since in training we the input is a sequence whereas the output is a single 
        # bin in the middle of the sequence. However, when trying to do genome-wide prediction, 
        # we need the output shape to match the input shape.

        # Warning: this code assumes that the final layer of the conv net is a 'valid' conv with 
        # filter_length = seq_length.
        num_bins = signalX.shape[0]

        # Initially, the shape of signalX is num_bins x num_input_marks.
        # We add (seq_length - 1) / 2 zeroes to both sides of the input, so that the 
        # resulting shape of the padded input is (num_bins + seq_length - 1) x num_input_marks.
        # The shape of the output will then be exactly num_bins x num_input_marks.
        assert len(signalX.shape) == 2
        assert signalX.shape[1] == self.num_input_marks
        signalX_pad = pad_sequence_with_zeros(
            signalX,
            padding=(self.dataset_params['seq_length'] - 1) // 2)

        # After padding, we reshape the input to fit the Keras predict() API, 
        # which requires a 3-tensor where the first dimension is the number of examples.
        # In our case, the number of examples is always 1 when doing genome-wide prediction.
        signalX = np.reshape(
            signalX_pad,
            [1, signalX_pad.shape[0], signalX_pad.shape[1]])

        if self.model_library == 'keras':
            Y = self.model.predict(signalX)
        elif self.model_library == 'pytorch':
            if device is None:
                device = DEVICE
            self.model = self.model.to(device)
            self.model.eval()
            with torch.no_grad():
                X = torch.from_numpy(signalX).float().to(device)
                Y = self.model(X).detach().cpu().numpy()

        Y = Y[0]

        assert Y.shape[0] == num_bins
        assert Y.shape[1] == self.num_output_marks

        return Y




class SeqToSeq(SeqModel):

    def __init__(self, model_params):
        """
        Initializes the correct model based on model_params.
        """

        super(SeqToSeq, self).__init__(model_params)

        if model_params['model_type'] == 'lstm':
            predict_binary_output = model_params['predict_binary_output']
            hidden_size = model_params['hidden_size']
            num_layers = model_params['num_layers']
            bidirectional = model_params['bidirectional']
            dropout = model_params['dropout']

            model = LSTMModel(
                predict_binary_output=predict_binary_output,
                in_channels=self.num_input_marks,
                out_channels=self.num_output_marks,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=bidirectional,
                p_dropout=dropout
            )
        elif model_params['model_type'] == 'encoder-decoder':
            predict_binary_output = model_params['predict_binary_output']
            hidden_size = model_params['hidden_size']
            num_layers = model_params['num_layers']
            bidirectional = model_params['bidirectional']
            dropout = model_params['dropout']
            teacher_forcing = model_params['teacher_forcing']
            # seq_length = self.dataset_params['seq_length']

            model = EncoderDecoder(
                predict_binary_output=predict_binary_output,
                in_channels=self.num_input_marks,
                out_channels=self.num_output_marks,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=bidirectional,
                p_dropout=dropout,
                teacher_forcing=teacher_forcing,
                # seq_length=seq_length
            )
        elif model_params['model_type'] == 'cnn-encoder-decoder':
            predict_binary_output = model_params['predict_binary_output']
            hidden_size = model_params['hidden_size']
            kernel_size = model_params['kernel_size']
            stride = model_params['stride']
            dilation = model_params['dilation']
            num_layers = model_params['num_layers']
            residual = model_params['residual']
            dropout = model_params['dropout']
            seq_length = self.dataset_params['seq_length']

            model = CnnEncoderDecoder(
                predict_binary_output=predict_binary_output,
                in_channels=self.num_input_marks,
                out_channels=self.num_output_marks,
                hidden_size=hidden_size,
                num_layers=num_layers,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                residual=residual,
                p_dropout=dropout,
                seq_length=seq_length,
                seq2seq=True
            )

        elif model_params['model_type'] == 'adv-cnn-encoder-decoder':
            predict_binary_output = model_params['predict_binary_output']
            hidden_size = model_params['hidden_size']
            kernel_size = model_params['kernel_size']
            stride = model_params['stride']
            dilation = model_params['dilation']
            num_layers = model_params['num_layers']
            residual = model_params['residual']
            dropout = model_params['dropout']
            seq_length = self.dataset_params['seq_length']

            disc_hidden_size = model_params['disc_hidden_size'] 
            disc_num_layers = model_params['disc_num_layers']
            disc_kernel_size = model_params['disc_kernel_size']
            disc_dilation = model_params['disc_dilation']            
            
            model = AdvCnnEncoderDecoder(
                predict_binary_output=predict_binary_output,
                in_channels=self.num_input_marks,
                out_channels=self.num_output_marks,
                hidden_size=hidden_size,
                num_layers=num_layers,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                residual=residual,
                p_dropout=dropout,
                seq_length=seq_length, 
                seq2seq=True,
                disc_hidden_size=disc_hidden_size, 
                disc_num_layers=disc_num_layers,
                disc_kernel_size=disc_kernel_size, 
                disc_dilation=disc_dilation
            )
        else:
            raise Exception("Model type not recognized")

        self.model = model
        self.save_model_params()

    def process_X(self, X):
        """
        See documentation in SeqModel.
        Takes in a matrix X of shape num_examples x seq_length x num_histone_marks,
        returned from extractDataset.load_seq_dataset, and processes it as necessary
        for the type of model. X should be the input data that is fed to the model.
        """

        return X

    def process_Y(self, Y):
        """
        See documentation in SeqModel.
        Takes in a matrix Y of shape num_examples x seq_length x num_histone_marks,
        returned from extractDataset.load_seq_dataset, and processes it as necessary
        for the type of model. Y should represent the desired output of the model.
        """

        return Y

    def predict_samples(self, signalX, device=None):
        """
        Takes in input signalX of whatever dimensions are needed for the model, which
        is subclass-dependent. It passes it through the model and returns the output matrix.
        """
        Y = self.SeqToX_predict_samples(signalX, device=device)

        return Y


    def predict_sequence(self, signalX, device=None):
        """
        Takes in input matrix signalX of dimensions num_bins x num_input_marks
        and passes it through the model,
        returning an output matrix of num_bins x num_output_marks.
        """
        # print(signalX.shape)
        # print(signalX[0], signalX[1])
        num_bins = signalX.shape[0]
        num_input_marks = signalX.shape[1]

        if self.model_library == 'keras':
            Y = self.model.predict(signalX)
        elif self.model_library == 'pytorch':
            if True:
                device = DEVICE
            self.model = self.model.to(device)
            if self.model_params['model_type'] == 'encoder-decoder':
                self.model.decoder.device = device
            self.model.eval()
            with torch.no_grad():
                if num_bins == 10000:
                    #X = torch.from_numpy(signalX).float().view(10, 1000, num_input_marks).to(device)
                    X = torch.from_numpy(signalX).float().view(-1, num_bins, num_input_marks).to(device)
                else:
                    #print("num_bins", num_bins)
                    X = torch.from_numpy(signalX).float().view(-1, num_bins, num_input_marks).to(device)
                Y = self.model(X).detach().cpu().view(-1, num_bins, self.num_output_marks).numpy()
                Y = Y[0]
                return Y
                
                if (num_bins % 1000) == 0:
                    # Y = torch.zeros((num_bins, self.num_output_marks)).numpy()
                    # return Y
                    pass
                else:
                    print("num_bins", num_bins)
                if (num_bins % 1000) == 0:
                    start_X = torch.from_numpy(signalX).float().view(-1, 1000, num_input_marks).to(device)                    
                else:
                    start_X = torch.from_numpy(signalX[:-(num_bins % 1000)]).float().view(-1, 1000, num_input_marks).to(device)
                    end_X = torch.from_numpy(signalX[-(num_bins % 1000):]).float().view(-1, (num_bins % 1000), num_input_marks).to(device)
                # print(X[0][0], X[0][1])
                # X = torch.from_numpy(signalX).float().view(-1, num_bins, num_input_marks).to(device)
                start_Y = self.model(start_X).detach().cpu().view(1, -1, self.num_output_marks).numpy()
                if (num_bins % 1000) != 0:
                    end_Y = self.model(end_X).detach().cpu().view(1, -1, self.num_output_marks).numpy()

        # print("signalX ", signalX.shape)
        # print("Y ", Y.shape, num_bins, self.num_output_marks)
        start_Y = start_Y[0]
        if (num_bins % 1000) != 0:
            end_Y = end_Y[0]
        if (num_bins % 1000) != 0:
            Y = np.append(start_Y, end_Y, axis=0)
        else:
            Y = start_Y
            
        assert Y.shape[0] == num_bins
        assert Y.shape[1] == self.num_output_marks

        return Y



