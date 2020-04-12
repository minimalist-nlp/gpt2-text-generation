# -*- coding: utf-8 -*-
import logging as log
import pdb
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, RandomSampler
from transformers import GPT2Tokenizer, GPT2LMHeadModel

import pytorch_lightning as pl
from gpt2_tokenizer import GPT2TextEncoder
from dataloader import text_dataset
from test_tube import HyperOptArgumentParser
from torchnlp.encoders import LabelEncoder
from torchnlp.utils import collate_tensors


class GPT2LanguageModel(pl.LightningModule):
    """
    Sample model to show how to train GPT2 with a Language Model head.
    
    :param hparams: ArgumentParser containing the hyperparameters.
    """

    def __init__(self, hparams) -> None:
        super(GPT2LanguageModel, self).__init__()
        self.hparams = hparams
        self.batch_size = hparams.batch_size
        self.output_units = 768 #self.gpt2.state_dict()["ln_f.bias"].shape[0] --> bias from last layer of the GPT2 model
        # build model
        self.__build_model()

        # Loss criterion initialization.
        self.__build_loss()


    def __build_model(self) -> None:
        """ Init GPT2 model + tokenizer + language model head."""
        self.gpt2 = GPT2LMHeadModel.from_pretrained(
            "gpt2", output_hidden_states=True
        )
        # Tokenizer
        self.tokenizer = GPT2TextEncoder("gpt2")
        self.gpt2.resize_token_embeddings(len(self.tokenizer.tokenizer))

    def __build_loss(self):
        """ Initializes the loss function/s. """
        self._loss = nn.CrossEntropyLoss(ignore_index=self.tokenizer.padding_index)

    def unfreeze_encoder(self) -> None:
        """ un-freezes the encoder layer. """
        if self._frozen:
            log.info(f"\n-- Encoder model fine-tuning")
            for param in self.gpt2.parameters():
                param.requires_grad = True

    def freeze_encoder(self) -> None:
        """ freezes the encoder layer. """
        for param in self.gpt2.parameters():
            param.requires_grad = False

    def generate(self, sample: dict) -> dict:
        """ Predict function.
        :param sample: dictionary with the text we want to classify.

        Returns:
            Dictionary with the input text and the predicted label.
        """
        if self.training:
            self.eval()

        with torch.no_grad():
            input_seq = sample["text"]
            inputs = self.tokenizer.encode(input_seq)
            bos_tokens = torch.full([1], self.tokenizer.stoi["<|endoftext|>"], dtype=torch.long)
            shifted_input = torch.cat((bos_tokens, inputs))
            trg_mask = (shifted_input[:len(inputs)+1] != self.tokenizer.padding_index).unsqueeze(1)
            output_seq = shifted_input[:len(inputs)+1]
            k = 1
            predicted_token = torch.Tensor([0])
            while predicted_token.unsqueeze(-1)[0] != self.tokenizer.padding_index:
                outputs = self.forward(output_seq)
                lm_logits = outputs["lm_logits"]
                logits = lm_logits[-1, :]
                predicted_token = logits.max(-1)[1]
                output_seq = torch.cat([output_seq, predicted_token.unsqueeze(-1)])
                k+=1       
            output_seq = output_seq[1:-1]
            output_sentence = self.tokenizer.decode(output_seq)
            print(output_sentence)

        return output_sentence

    def forward(self, tokens):
        """ Usual pytorch forward function.
        :param tokens: text sequences [batch_size x src_seq_len]

        Returns:
            Dictionary with model outputs (e.g: logits)
        """
        gpt2_outputs = self.gpt2(tokens)
        lm_logits = gpt2_outputs[0]

        return {"lm_logits": lm_logits}

    def loss(self, predictions: dict, labels: dict) -> torch.tensor:
        """
        Computes Causal Language Modelling (CLM) Loss value according to a loss function.
        :param predictions: model specific output. Must contain a key 'lm_logits' with
            a tensor [batch_size x 1] with model predictions
        :param labels: Labels for language modeling.

        Returns:
            torch.tensor with loss value.
        """
        batch_logits = predictions["lm_logits"][..., :-1, :].contiguous()
        target_labels = labels["tokens"][..., 1:].contiguous()
        loss = self._loss(batch_logits.view(-1, batch_logits.size(-1)), target_labels.view(-1))
        return loss

    def prepare_sample(self, sample: list) -> (dict):
        """
        Function that prepares a sample to input the model.
        :param sample: list of dictionaries.
        
        Returns:
            - dictionary with the model inputs.
        """
        sample = collate_tensors(sample)
        tokens, lengths = self.tokenizer.batch_encode(sample["text"])
        inputs = {"tokens": tokens}

        return inputs

    def training_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        """ 
        Runs one training step. This usually consists in the forward function followed
            by the loss function.
        
        :param batch: The output of your dataloader. 
        :param batch_nb: Integer displaying which batch this is

        Returns:
            - dictionary containing the loss and the metrics to be added to the lightning logger.
        """
        inputs = batch
        model_out = self.forward(**inputs)
        loss_val = self.loss(model_out, inputs)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_val = loss_val.unsqueeze(0)

        tqdm_dict = {"train_loss": loss_val}
        output = OrderedDict(
            {"loss": loss_val, "progress_bar": tqdm_dict, "log": tqdm_dict}
        )

        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        """ Similar to the training step but with the model in eval mode.

        Returns:
            - dictionary passed to the validation_end function.
        """
        inputs = batch
        model_out = self.forward(**inputs)
        loss_val = self.loss(model_out, inputs)

        if self.on_gpu:
            loss_val = loss_val.cuda(loss_val.device.index)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_val = loss_val.unsqueeze(0)

        output = OrderedDict({"val_loss": loss_val})

        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_end(self, outputs: list) -> dict:
        """ Function that takes as input a list of dictionaries returned by the validation_step
        function and measures the model performance accross the entire validation set.
        
        Returns:
            - Dictionary with metrics to be added to the lightning logger.  
        """
        val_loss_mean = 0
        for output in outputs:
            val_loss = output["val_loss"]
            # reduce manually when using dp
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_loss = torch.mean(val_loss)
            val_loss_mean += val_loss
        val_loss_mean /= len(outputs)
        perplexity = torch.exp(val_loss_mean.clone().detach())
        tqdm_dict = {"val_loss": val_loss_mean, "perplexity": perplexity}
        result = {
            "progress_bar": tqdm_dict,
            "log": tqdm_dict,
            "perplexity": perplexity
        }
        return result

    def configure_optimizers(self):
        """ Sets Learning rate for different parameter groups. """
        parameters = [
            {
                "params": self.gpt2.parameters(),
                "lr": self.hparams.learning_rate,
            },
        ]
        optimizer = optim.Adam(parameters, lr=self.hparams.learning_rate)
        return [optimizer], []


    def __retrieve_dataset(self, train=True, val=True, test=True):
        """ Retrieves task specific dataset """
        return text_dataset(self.hparams, train, val, test)

    @pl.data_loader
    def train_dataloader(self) -> DataLoader:
        """ Function that loads the train set. """
        self._train_dataset = self.__retrieve_dataset(val=False, test=False)[0]
        return DataLoader(
            dataset=self._train_dataset,
            sampler=RandomSampler(self._train_dataset),
            batch_size=self.hparams.batch_size,
            collate_fn=self.prepare_sample,
            num_workers=self.hparams.loader_workers,
        )

    @pl.data_loader
    def val_dataloader(self) -> DataLoader:
        """ Function that loads the validation set. """
        self._dev_dataset = self.__retrieve_dataset(train=False, test=False)[0]
        return DataLoader(
            dataset=self._dev_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.prepare_sample,
            num_workers=self.hparams.loader_workers,
        )

    @pl.data_loader
    def test_dataloader(self) -> DataLoader:
        """ Function that loads the validation set. """
        self._test_dataset = self.__retrieve_dataset(train=False, val=False)[0]
        return DataLoader(
            dataset=self._test_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.prepare_sample,
            num_workers=self.hparams.loader_workers,
        )

    @classmethod
    def add_model_specific_args(
        cls, parser: HyperOptArgumentParser
    ) -> HyperOptArgumentParser:
        """ Parser for Estimator specific arguments/hyperparameters. 
        :param parser: HyperOptArgumentParser obj

        Returns:
            - updated parser
        """
        parser.add_argument(
            "--learning_rate",
            default=3e-05,
            type=float,
            help="Learning rate.",
        )
        # Data Args:
        parser.add_argument(
            "--train_csv",
            default="data/train_data.csv",
            type=str,
            help="Path to the file containing the train data.",
        )
        parser.add_argument(
            "--dev_csv",
            default="data/valid_data.csv",
            type=str,
            help="Path to the file containing the dev data.",
        )
        parser.add_argument(
            "--test_csv",
            default="data/valid_data.csv",
            type=str,
            help="Path to the file containing the dev data.",
        )
        parser.add_argument(
            "--loader_workers",
            default=8,
            type=int,
            help="How many subprocesses to use for data loading. 0 means that \
                the data will be loaded in the main process.",
        )
        return parser
