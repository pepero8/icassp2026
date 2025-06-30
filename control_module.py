from torch import nn
import torch
import torch.nn.functional as F

from cross_attention_transformer import CrossAttention
from transformer_encoder import TransformerEncoder


class ControlModule(nn.Module):
    def __init__(self, config, num_speakers):
        super(ControlModule, self).__init__()
        self.config = config
        self.transformer_encoder = TransformerEncoder(
            **config.transformer_encoder  # ? Pretrained transformer encoder에 해당
        )
        self.cross_attention = CrossAttention(
            query_dim=self.transformer_encoder.dim,
            key_dim=self.transformer_encoder.dim
            + config.addressee_predictor.hidden_dim,
            value_dim=self.transformer_encoder.dim
            + config.addressee_predictor.hidden_dim,
            **config.cross_attention,
        )
        self.conv_pool = nn.Conv1d(
            # config.conv_pool.feature_dim,
            in_channels=self.transformer_encoder.dim
            + config.addressee_predictor.hidden_dim,
            out_channels=config.conv_pool.out_channels,
            kernel_size=config.conv_pool.kernel_size,
            stride=config.conv_pool.stride,
            padding=config.conv_pool.padding,
        )

        # self.addressee_predictor = nn.Linear(
        #     self.transformer_encoder.dim, config.num_speakers
        # )
        self.addressee_predictor_hidden = nn.Sequential(
            nn.Linear(
                config.conv_pool.out_channels + self.transformer_encoder.dim,
                config.addressee_predictor.hidden_dim,
            ),
            nn.ReLU(),
        )
        self.addressee_predictor_out = nn.Linear(
            config.addressee_predictor.hidden_dim,
            num_speakers + 2,  # + 2 for 'assistant' and 'all'
        )

        self.control_predictor_attention = CrossAttention(
            query_dim=self.transformer_encoder.dim
            + config.addressee_predictor.hidden_dim,
            key_dim=self.transformer_encoder.dim
            + config.addressee_predictor.hidden_dim,
            value_dim=self.transformer_encoder.dim
            + config.addressee_predictor.hidden_dim,
            **config.control_predictor.cross_attention,
        )  # ? Control net에 해당

        self.control_predictor_linear = nn.Linear(
            self.transformer_encoder.dim + config.addressee_predictor.hidden_dim, 4
        )
        self.ai_addressee_predictor_hidden = nn.Sequential(
            nn.Linear(
                self.transformer_encoder.dim + config.addressee_predictor.hidden_dim,
                config.addressee_predictor.hidden_dim,
            ),
            nn.ReLU(),
        )
        self.ai_addressee_predictor_linear = nn.Linear(
            config.addressee_predictor.hidden_dim, num_speakers + 1  # + 1 for 'NA'
        )

    def forward(self, x, dialog_memory):
        """
        x: (1, T), Speaker and content tokens
        dialog_memory: (L, D), D is dim of cls token(dim of transformer encoder output)
        """

        cls = self.transformer_encoder(x)  # (1, D)

        if dialog_memory.numel() != 0:
            out_dialog, _ = self.cross_attention(
                query=cls.unsqueeze(1),  # cls.unsqueeze: (1, 1, D)
                key=dialog_memory.unsqueeze(0),  # (1, L, D+hidden_dim)
                value=dialog_memory.unsqueeze(0),  # (1, L, D+hidden_dim)
            )  # (1, L, D+hidden_dim)
        else:
            zero_addressee_emb = torch.zeros(
                (1, self.config.addressee_predictor.hidden_dim), device=x.device
            )
            out_dialog = torch.cat((cls, zero_addressee_emb), dim=-1).unsqueeze(
                1
            )  # (1, 1, D+hidden_dim)

        out = out_dialog.transpose(1, 2)  # (1, D+hidden_dim, L)
        out = self.conv_pool(out)  # (1, 512, L)
        out = F.adaptive_avg_pool1d(out, 1).squeeze(
            -1
        )  # Global average pooling. (1, 512)

        out = torch.concat((out, cls), dim=-1)  # (1, 512 + D)
        addressee_emb = self.addressee_predictor_hidden(out)  # (1, hidden_dim)
        addressee = self.addressee_predictor_out(
            addressee_emb
        )  # raw logits(linear의 output), (1, num_speakers)

        if dialog_memory.numel() != 0:
            out, _ = self.control_predictor_attention(
                # query=cls.unsqueeze(1),  # (1, 1, D)
                query=torch.cat((cls, addressee_emb), dim=-1).unsqueeze(
                    1
                ),  # (1, 1, D+hidden_dim)
                key=dialog_memory.unsqueeze(0),  # (1, L, D+hidden_dim)
                # key=out_dialog,
                value=dialog_memory.unsqueeze(0),  # (1, L, D+hidden_dim)
                # value=out_dialog,
            )  # (1, L, D+hidden_dim)
        else:
            out = out_dialog  # (1, 1, D+hidden_dim)
            # out = cls.unsqueeze(1)  # (1, 1, D)

        out = out.transpose(1, 2)  # (1, D+hidden_dim, L)
        # todo: make out shape (1, D+hidden_dim)
        out = F.adaptive_avg_pool1d(out, 1).squeeze(
            -1
        )  # Global average pooling. (1, D+hidden_dim)

        control_token = self.control_predictor_linear(
            out
        )  # raw logits(linear의 output), (1, 4)
        ai_addressee_emb = self.ai_addressee_predictor_hidden(out)
        ai_addressee = self.ai_addressee_predictor_linear(
            ai_addressee_emb
        )  # (1, num_speakers + 1)

        return (
            addressee,
            addressee_emb,
            ai_addressee,
            ai_addressee_emb,
            control_token,
            cls,
        )

    def inference(self, x, dialog_memory):
        pass
