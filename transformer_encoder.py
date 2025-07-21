import torch
from torch import nn
from transformers import BertModel, BertTokenizer


class TransformerEncoder(nn.Module):
    def __init__(self, pretrained_transformer="bert"):
        super(TransformerEncoder, self).__init__()

        # todo: Initialize transformer model -> need test
        if pretrained_transformer == "bert":
            # Initialize BERT model
            self.__model = BertModel.from_pretrained("bert-base-uncased")
            self.__tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

            special_tokens = [
                "<Speaker_A>",
                "<Speaker_B>",
                "<Speaker_C>",
                "<Speaker_D>",
                "<silence>",
            ]
            self.__tokenizer.add_tokens(special_tokens)
            self.__model.resize_token_embeddings(len(self.__tokenizer))

            for param in self.__model.parameters():
                param.requires_grad = False

            # Unfreeze embedding layer
            for param in self.__model.embeddings.parameters():
                param.requires_grad = True

    @property
    def dim(self):
        return self.__model.config.hidden_size

    @property
    def tokenizer(self):
        return self.__tokenizer

    @property
    def device(self):
        return self.__model.device

    def forward(self, x):
        """
        x: (1, T), token sequence of shape (batch_size, sequence_length)
        """

        x = x.to(self.__model.device)

        # > return the first token([CLS]) of output sequence
        # outputs = self.__model(inputs_embeds=x)  # (1, T, D)
        outputs = self.__model(input_ids=x)  # (1, T, D)
        # cls = outputs.last_hidden_state[:, -1, :] # last token
        cls = outputs.last_hidden_state[:, 0, :]  # first token

        return cls  # (1, D)

    def inference(self, x):
        pass
