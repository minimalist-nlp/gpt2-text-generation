# -*- coding: utf-8 -*-
import torch
from transformers import GPT2Tokenizer

from torchnlp.encoders import Encoder
from torchnlp.encoders.text import stack_and_pad_tensors
from torchnlp.encoders.text.text_encoder import TextEncoder


class GPT2TextEncoder(TextEncoder):
    """
    Wrapper arround GPT2 tokenizer.
    """

    def __init__(self, pretrained_model) -> None:
        self.enforce_reversible = False
        self.tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)
        self.stoi = self.tokenizer.encoder
        self.itos = self.tokenizer.decoder
        special_tokens_dict = {'pad_token': '<PAD>'}
        num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
        print('We have added', num_added_toks, 'tokens')
        # num_added_toks = self.tokenizer.add_tokens(['<END-VERSE>'])
        # print('We have added', num_added_toks, 'tokens')


    @property
    def unk_index(self) -> int:
        """ Returns the index used for the unknown token. """
        return self.tokenizer.unk_token_id

    @property
    def bos_index(self) -> int:
        """ Returns the index used for the begin-of-sentence token. """
        return self.tokenizer.cls_token_id

    @property
    def eos_index(self) -> int:
        """ Returns the index used for the end-of-sentence token. """
        return self.tokenizer.sep_token_id

    @property
    def padding_index(self) -> int:
        """ Returns the index used for padding. """
        return self.tokenizer.pad_token_id

    @property
    def vocab(self) -> list:
        """
        Returns:
            list: List of tokens in the dictionary.
        """
        return self.tokenizer.vocab

    @property
    def vocab_size(self) -> int:
        """
        Returns:
            int: Number of tokens in the dictionary.
        """
        return len(self.itos)

    def encode(self, sequence: str) -> torch.Tensor:
        """ Encodes a 'sequence'.
        :param sequence: String 'sequence' to encode.
        
        Returns:
            - torch.Tensor: Encoding of the 'sequence'.
        """
        sequence = TextEncoder.encode(self, sequence)
        vector = self.tokenizer.encode(sequence)
        return torch.tensor(vector)

    def batch_encode(self, iterator, dim=0, **kwargs) -> (torch.Tensor, torch.Tensor):
        """
        :param iterator (iterator): Batch of text to encode.
        :param dim (int, optional): Dimension along which to concatenate tensors.
        :param **kwargs: Keyword arguments passed to 'encode'.
            
        Returns
            torch.Tensor, torch.Tensor: Encoded and padded batch of sequences; Original lengths of
                sequences.
        """
        return stack_and_pad_tensors(
            Encoder.batch_encode(self, iterator, **kwargs),
            padding_index=self.padding_index,
            dim=dim,
        )
    
    def decode(self, embeddings):
        """ Encodes a 'sequence'.
        Requires a space to start the input string  -> the encoding methods should be called with the 
        'add_prefix_space' flag set to 'True'. Otherwise, this tokenizer encode and decode will not conserve the
        absence of space at the beginning of a string.

        :param sequence: String 'sequence' to encode.
        
        Returns:
            - torch.Tensor: Encoding of the 'sequence'.
        """
        sequence = TextEncoder.decode(self, embeddings)
        vector = self.tokenizer.decode(embeddings)
        return vector
