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

        self.addressee_embedding = nn.Embedding(
            num_embeddings=self.config.num_speakers + 2,  # +2 for 'assistant' and 'all'
            embedding_dim=self.config.control_module.addressee_predictor.hidden_dim,
        )  # (num_speakers+2, hidden_dim)

    def forward(
        self,
        x,
        addressee_to_idx=None,
        ai_addressee_to_idx=None,
        control_token_to_idx=None,
        mode=None,
    ):
        """
        x: Chunk instance
        """
        # audio = x.audio  # 1sec audio chunk(float32 tensor), (1, 22050)
        # out = self.sa_asr.diarize(audio)  # (1, D, T)

        token_sequence = self.tokenizer.encode(
            x.tape, add_special_tokens=True, return_tensors="pt"
        )  # (1, T) add_special_tokens=True로 주면 문장 처음에 [CLS] 토큰, 마지막에 [SEP] 토큰이 추가됨

        target_addressee = x.addressee  # (1, ) tensor of addressee label
        target_ai_addressee = x.ai_addressee  # (1, ) tensor
        # target_control_token = x.control_token  # (1, ) tensor of control token label
        target_addressee_idx = (
            addressee_to_idx[target_addressee] if addressee_to_idx else None
        )
        target_ai_addressee_idx = (
            ai_addressee_to_idx[target_ai_addressee] if ai_addressee_to_idx else None
        )
        # target_control_token_idx = control_token_to_idx[target_control_token] if control_token_to_idx else None

        (
            addressee,
            addressee_embd,
            # ai_addressee,
            # ai_addressee_embd,
            # control_token,
            cls,
        ) = self.control_module(token_sequence, self.dialog_memory)

        if mode == "train":
            addressee_label = (
                torch.tensor(target_addressee_idx).unsqueeze(0).to(cls.device)
            )  # (1, )
            ai_addressee_label = (
                torch.tensor(target_ai_addressee_idx).unsqueeze(0).to(cls.device)
            )  # (1, )
            # # control_token_label = target_control_token_idx.unsqueeze(0)  # (1, )
            addressee_embd = self.addressee_embedding(
                addressee_label
            )  # (1, hidden_dim)
            ai_addressee_embd = self.addressee_embedding(ai_addressee_label)

        else:
            addressee_label = addressee.argmax(dim=1)
            # ai_addressee_label = ai_addressee.argmax(dim=1)
            ai_addressee_label = (
                torch.tensor(target_ai_addressee_idx).unsqueeze(0).to(cls.device)
            )  # (1, )
            addressee_embd = self.addressee_embedding(
                addressee_label
            )  # (1, hidden_dim)
            ai_addressee_embd = self.addressee_embedding(
                ai_addressee_label
            )  # (1, hidden_dim)

        # > update dialog memory with cls token and addressee embedding
        new_token = torch.cat(
            (addressee_embd.detach(), cls.detach()),
            dim=1,  # tensors should be detached to avoid gradients flowing into dialog memory
        )  # (1, D+hidden_dim), D is dim of cls token, hidden_dim is dim of addressee embedding
        if self.dialog_memory.numel() == 0:
            self.dialog_memory = new_token
        else:
            self.dialog_memory = torch.cat((self.dialog_memory, new_token), dim=0)

        ##############################################
        # AI response가 존재하는지 아닌지에 대한 check 필요
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
        ##############################################

        return addressee  # , ai_addressee, control_token

    def reset_dialog_memory(self):
        """
        Reset dialog memory to initial state.
        """
        self.dialog_memory = torch.Tensor([])
        self.dialog_memory.requires_grad = False

    def inference(self, x):
        pass
