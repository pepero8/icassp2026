import torch
from torch import nn

from control_module import ControlModule

# from sa_asr_wrapper import SAASRWrapper


class SAASRControl(nn.Module):
    def __init__(self, config):
        super(SAASRControl, self).__init__()
        self.config = config
        # self.sa_asr: SAASRWrapper = None
        self.control_module = ControlModule(config.control_module, config.num_speakers)
        self.dialog_memory = torch.Tensor([])  # (L, D)

        self.tokenizer = self.control_module.transformer_encoder.tokenizer

        self.dialog_memory.requires_grad = False

    def forward(self, x):
        """
        x: Chunk instance
        """
        # audio = x.audio  # 1sec audio chunk(float32 tensor), (1, 22050)
        # out = self.sa_asr.diarize(audio)  # (1, D, T)

        token_sequence = self.tokenizer.encode(
            x.tape, add_special_tokens=True, return_tensors="pt"
        )  # (1, T) add_special_tokens=True로 주면 문장 처음에 [CLS] 토큰, 마지막에 [SEP] 토큰이 추가됨

        (
            addressee,
            addressee_embd,
            ai_addressee,
            ai_addressee_embd,
            control_token,
            cls,
        ) = self.control_module(token_sequence, self.dialog_memory)

        # > update dialog memory with cls token and addressee embedding
        new_token = torch.cat(
            (addressee_embd.detach(), cls.detach()), dim=1 # tensors should be detached to avoid gradients flowing into dialog memory
        )  # (1, D+hidden_dim), D is dim of cls token, hidden_dim is dim of addressee embedding
        if self.dialog_memory.numel() == 0:
            self.dialog_memory = new_token
        else:
            self.dialog_memory = torch.cat((self.dialog_memory, new_token), dim=0)

        # > if ai response exists, append it to dialog memory
        if x.ai_response is not None:
            with torch.no_grad():
                ai_response_token_sequence = self.tokenizer.encode(
                    x.ai_response, add_special_tokens=True, return_tensors="pt"
                )

                # > pass it to transformer encoder to get cls token
                ai_response_cls = self.control_module.transformer_encoder(
                    ai_response_token_sequence
                )  # (1, D)

            new_token = torch.cat(
                (ai_addressee_embd.detach(), ai_response_cls.detach()), dim=1
            )  # (1, D+hidden_dim)

            self.dialog_memory = torch.cat((self.dialog_memory, new_token), dim=0)

        return addressee, ai_addressee, control_token

    def reset_dialog_memory(self):
        """
        Reset dialog memory to initial state.
        """
        self.dialog_memory = torch.Tensor([])
        self.dialog_memory.requires_grad = False

    def inference(self, x):
        pass
