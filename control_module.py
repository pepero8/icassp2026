from torch import dropout, nn
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
            + config.addressee_predictor.hidden_dim * 2,
            value_dim=self.transformer_encoder.dim
            + config.addressee_predictor.hidden_dim * 2,
            **config.cross_attention,
        )

        self.cls_transformer_attention = CrossAttention(
            query_dim=config.conv_pool.out_channels,
            key_dim=config.conv_pool.out_channels,
            value_dim=config.conv_pool.out_channels,
            embed_dim=config.conv_pool.out_channels,
            num_heads=1,
            dropout=0.1,
            bias=True,
        )

        self.conv_pool = nn.Conv1d(
            # config.conv_pool.feature_dim,
            in_channels=self.transformer_encoder.dim
            + config.addressee_predictor.hidden_dim * 2,
            out_channels=config.conv_pool.out_channels,
            kernel_size=config.conv_pool.kernel_size,
            stride=config.conv_pool.stride,
            padding=config.conv_pool.padding,
        )

        self.cls_conv_pool = nn.Conv1d(
            in_channels=self.transformer_encoder.dim,
            out_channels=config.conv_pool.out_channels,
            kernel_size=config.conv_pool.kernel_size,
            stride=config.conv_pool.stride,
            padding=config.conv_pool.padding,
        )

        self.transformer_layer = nn.Sequential()
        for i in range(config.transformer_layer.num_layers):
            self.transformer_layer.add_module(
                f"transformer_layer_{i}",
                nn.TransformerEncoderLayer(
                    d_model=config.conv_pool.out_channels,
                    nhead=config.transformer_layer.nhead,
                    dim_feedforward=config.transformer_layer.dim_feedforward,
                    dropout=config.transformer_layer.dropout,
                    activation=config.transformer_layer.activation,
                    batch_first=True,
                ),
            )

        # self.addressee_predictor = nn.Linear(
        #     self.transformer_encoder.dim, config.num_speakers
        # )
        self.addressee_predictor_hidden = nn.Sequential(
            nn.Linear(
                # config.conv_pool.out_channels + self.transformer_encoder.dim,
                config.conv_pool.out_channels * 2,
                config.addressee_predictor.hidden_dim,
            ),
            nn.ReLU(),
        )
        self.addressee_predictor_out = nn.Linear(
            config.addressee_predictor.hidden_dim,
            num_speakers + 2,  # + 2 for 'assistant' and 'all'
        )

        self.control_predictor_attention = CrossAttention(
            query_dim=config.addressee_predictor.hidden_dim
            + config.conv_pool.out_channels,
            key_dim=self.transformer_encoder.dim
            + config.addressee_predictor.hidden_dim * 2,
            value_dim=self.transformer_encoder.dim
            + config.addressee_predictor.hidden_dim * 2,
            **config.control_predictor.cross_attention,
        )  # ? Control net에 해당

        self.control_predictor_linear = nn.Linear(
            self.transformer_encoder.dim + config.addressee_predictor.hidden_dim * 2, 4
        )
        self.ai_addressee_predictor_hidden = nn.Sequential(
            nn.Linear(
                self.transformer_encoder.dim
                + config.addressee_predictor.hidden_dim * 2,
                config.addressee_predictor.hidden_dim,
            ),
            nn.ReLU(),
        )
        self.ai_addressee_predictor_linear = nn.Linear(
            config.addressee_predictor.hidden_dim,
            num_speakers + 2,  # + 2 for 'NA' and 'All'
        )

    def forward(self, x, dialog_memory, gt_addressee_emb, addressee_embedding_table):
        """
        x: (1, T), Speaker and content tokens
        dialog_memory: (L, D), D is dim of cls token(dim of transformer encoder output)
        """

        # > Ensure x is on the same device as model
        x = x.to(self.transformer_encoder.device)

        cls = self.transformer_encoder(x)  # (1, D)

        if dialog_memory.numel() != 0:
            out_dialog, _ = self.cross_attention(
                query=cls.unsqueeze(1),  # cls.unsqueeze: (1, 1, D)
                key=dialog_memory.unsqueeze(0),  # (1, L, D + 2*hidden_dim)
                value=dialog_memory.unsqueeze(0),  # (1, L, D + 2*hidden_dim)
            )  # (1, L, D + 2*hidden_dim)
        else:
            zero_addressee_emb = torch.zeros(
                (1, self.config.addressee_predictor.hidden_dim), device=x.device
            )
            zero_speaker_emb = torch.zeros(
                (1, self.config.addressee_predictor.hidden_dim), device=x.device
            )
            out_dialog = torch.cat(
                (zero_addressee_emb, cls, zero_speaker_emb), dim=-1
            ).unsqueeze(
                1
            )  # (1, 1, D + 2*hidden_dim)

        out = out_dialog.transpose(1, 2)  # (1, D + 2*hidden_dim, L)
        out = self.conv_pool(out)  # (1, 512, L)

        _cls = self.cls_conv_pool(cls.unsqueeze(-1))  # (1, 512, 1)
        _cls = _cls.squeeze(-1)  # (1, 512)

        # out = F.adaptive_avg_pool1d(out, 1).squeeze(
        #     -1
        # )  # Global average pooling. (1, 512)

        # > add _cls and out before passing to transformer layer
        out = out + _cls.unsqueeze(-1).expand(-1, -1, out.size(-1))  # (1, 512, L)

        # ========================== transformer layer로 변경 ==========================
        # > Add all-zero 512 dim embedding at the end of 'out'
        # out = F.pad(out, (0, 1, 0, 0), value=0)  #  (1, 512, L)
        out = out.transpose(1, 2)  # (1, L, 512)
        out = self.transformer_layer(out)  # (1, L, 512)
        # out = out.transpose(1, 2)  # (1, 512, L)
        # out = out[:, :, -1]  # (1, 512)
        # ========================== transformer layer로 변경 ==========================

        out, _ = self.cls_transformer_attention(
            query=_cls.unsqueeze(1),  # _cls.unsqueeze: (1, 1, 512)
            key=out,  # (1, L, 512)
            value=out,  # (1, L, 512)
        )  # (1, L, 512)
        out = out.sum(dim=1)  # (1, 512) # > sum along feaure length dimension

        out_for_addr_emb = torch.concat((out, _cls), dim=-1)  # (1, 512 + 512)
        _addressee_emb = self.addressee_predictor_hidden(
            out_for_addr_emb
        )  # (1, hidden_dim)
        addressee = self.addressee_predictor_out(
            _addressee_emb
        )  # raw logits(linear의 output), (1, num_speakers + 2)

        # addressee_label = addressee.argmax(dim=1)
        # addressee_emb = addressee_embedding_table(addressee_label)  # (1, hidden_dim)

        if dialog_memory.numel() != 0:
            out, _ = self.control_predictor_attention(
                # query=cls.unsqueeze(1),  # (1, 1, D)
                query=torch.cat((gt_addressee_emb, out), dim=-1).unsqueeze(
                    1
                ),  # (1, 1, hidden_dim+512)
                key=dialog_memory.unsqueeze(0),  # (1, L, D + 2*hidden_dim)
                # key=out_dialog,
                value=dialog_memory.unsqueeze(0),  # (1, L, D + 2*hidden_dim)
                # value=out_dialog,
            )  # (1, L, D + 2*hidden_dim)
        else:
            out = out_dialog  # (1, 1, D + 2*hidden_dim)
            # out = cls.unsqueeze(1)  # (1, 1, D)

        out = out.transpose(1, 2)  # (1, D + 2*hidden_dim, L)
        # todo: make out shape (1, D + 2*hidden_dim)
        # out = F.adaptive_avg_pool1d(out, 1).squeeze(
        #     -1
        # )  # Global average pooling. (1, D + 2*hidden_dim)
        out = out.sum(dim=-1)  # (1, D + 2*hidden_dim)

        control_token = self.control_predictor_linear(
            out
        )  # raw logits(linear의 output), (1, 4)
        ai_addressee_emb = self.ai_addressee_predictor_hidden(out)
        ai_addressee = self.ai_addressee_predictor_linear(
            ai_addressee_emb
        )  # (1, num_speakers + 2)

        return (
            addressee,
            gt_addressee_emb,
            ai_addressee,
            ai_addressee_emb,
            control_token,
            cls,
        )

    def inference(self, x, dialog_memory, addressee_embedding_table):
        """
        x: (1, T), Speaker and content tokens
        dialog_memory: (L, D), D is dim of cls token(dim of transformer encoder output)
        """

        # > Ensure x is on the same device as model
        x = x.to(self.transformer_encoder.device)

        cls = self.transformer_encoder(x)  # (1, D)

        if dialog_memory.numel() != 0:
            out_dialog, _ = self.cross_attention(
                query=cls.unsqueeze(1),  # cls.unsqueeze: (1, 1, D)
                key=dialog_memory.unsqueeze(0),  # (1, L, D + 2*hidden_dim)
                value=dialog_memory.unsqueeze(0),  # (1, L, D + 2*hidden_dim)
            )  # (1, L, D + 2*hidden_dim)
        else:
            zero_addressee_emb = torch.zeros(
                (1, self.config.addressee_predictor.hidden_dim), device=x.device
            )
            zero_speaker_emb = torch.zeros(
                (1, self.config.addressee_predictor.hidden_dim), device=x.device
            )
            out_dialog = torch.cat(
                (zero_addressee_emb, cls, zero_speaker_emb), dim=-1
            ).unsqueeze(
                1
            )  # (1, 1, D + 2*hidden_dim)

        out = out_dialog.transpose(1, 2)  # (1, D + 2*hidden_dim, L)
        out = self.conv_pool(out)  # (1, 512, L)

        _cls = self.cls_conv_pool(cls.unsqueeze(-1))  # (1, 512, 1)
        _cls = _cls.squeeze(-1)  # (1, 512)

        # out = F.adaptive_avg_pool1d(out, 1).squeeze(
        #     -1
        # )  # Global average pooling. (1, 512)

        # ========================== transformer layer로 변경 ==========================
        # > Add all-zero 512 dim embedding at the end of 'out'
        # out = F.pad(out, (0, 1, 0, 0), value=0)  #  (1, 512, L)

        out = out.transpose(1, 2)  # (1, L, 512)
        out = self.transformer_layer(out)  # (1, L, 512)
        # out = out.transpose(1, 2)  # (1, 512, L)
        # out = out[:, :, -1]  # (1, 512)
        # ========================== transformer layer로 변경 ==========================

        out, _ = self.cls_transformer_attention(
            query=_cls.unsqueeze(1),  # _cls.unsqueeze: (1, 1, 512)
            key=out,  # (1, L, 512)
            value=out,  # (1, L, 512)
        )  # (1, L, 512)
        out = out.sum(dim=1)  # (1, 512) # > sum along feaure length dimension

        out_for_addr_emb = torch.concat((out, _cls), dim=-1)  # (1, 512 + 512)
        _addressee_emb = self.addressee_predictor_hidden(
            out_for_addr_emb
        )  # (1, hidden_dim)
        addressee = self.addressee_predictor_out(
            _addressee_emb
        )  # raw logits(linear의 output), (1, num_speakers + 2)

        addressee_label = addressee.argmax(dim=1)
        addressee_emb = addressee_embedding_table(addressee_label)  # (1, hidden_dim)

        if dialog_memory.numel() != 0:
            out, _ = self.control_predictor_attention(
                # query=cls.unsqueeze(1),  # (1, 1, D)
                query=torch.cat((addressee_emb, out), dim=-1).unsqueeze(
                    1
                ),  # (1, 1, D+hidden_dim)
                key=dialog_memory.unsqueeze(0),  # (1, L, D + 2*hidden_dim)
                # key=out_dialog,
                value=dialog_memory.unsqueeze(0),  # (1, L, D + 2*hidden_dim)
                # value=out_dialog,
            )  # (1, L, D + 2*hidden_dim)
        else:
            out = out_dialog  # (1, 1, D + 2*hidden_dim)
            # out = cls.unsqueeze(1)  # (1, 1, D)

        out = out.transpose(1, 2)  # (1, D + 2*hidden_dim, L)
        # todo: make out shape (1, D + 2*hidden_dim)
        # out = F.adaptive_avg_pool1d(out, 1).squeeze(
        #     -1
        # )  # Global average pooling. (1, D + 2*hidden_dim)
        out = out.sum(dim=-1)  # (1, D + 2*hidden_dim)

        control_token = self.control_predictor_linear(
            out
        )  # raw logits(linear의 output), (1, 4)
        ai_addressee_emb = self.ai_addressee_predictor_hidden(out)
        ai_addressee = self.ai_addressee_predictor_linear(
            ai_addressee_emb
        )  # (1, num_speakers + 2)

        return (
            addressee,
            addressee_emb,
            ai_addressee,
            ai_addressee_emb,
            control_token,
            cls,
        )
