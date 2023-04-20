import copy
from typing import Optional, Any, Union, Callable

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.module import Module
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.container import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer
from torch.nn.modules.transformer import TransformerDecoder, TransformerDecoderLayer

class TemporalTransformer(Module):
    def __init__(self, opt) -> None:
        super(TemporalTransformer, self).__init__()
        nhead: int = 8
        num_encoder_layers: int = 6
        num_decoder_layers: int = 6
        dim_feedforward: int = 2048
        dropout: float = 0.1
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu
        layer_norm_eps: float = 1e-5
        batch_first: bool = True
        norm_first: bool = False,
        device=None
        dtype=None
        d_model = opt.v_f_len + opt.i_f_len
        factory_kwargs = {'device': device, 'dtype': dtype}

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                activation, layer_norm_eps, batch_first, norm_first,
                                                **factory_kwargs)
        encoder_norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TemporalTransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                activation, layer_norm_eps, batch_first, norm_first,
                                                **factory_kwargs)
        decoder_norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.decoder = TemporalTransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

        self.batch_first = batch_first

    def forward(self, src: Tensor, tgt: Tensor, history_out = None) -> Tensor:

        if not self.batch_first and src.size(1) != tgt.size(1):
            raise RuntimeError("the batch number of src and tgt must be equal")
        elif self.batch_first and src.size(0) != tgt.size(0):
            raise RuntimeError("the batch number of src and tgt must be equal")

        if src.size(2) != self.d_model or tgt.size(2) != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")

        memory = self.encoder(src, None, None)
        output = self.decoder(tgt, memory, history_out = history_out)

        return output

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

class TemporalTransformerDecoder(TransformerDecoder):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__(decoder_layer, num_layers, norm)
    
    def forward(self, tgt: Tensor, memory: Tensor, history_out: Tensor = None) -> Tensor:
        output = tgt

        for mod in self.layers:
            output = mod(output, memory, history_out)  # 传入history_out

        if self.norm is not None:
            output = self.norm(output)

        return output

class TemporalTransformerDecoderLayer(TransformerDecoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu, layer_norm_eps=0.00001, batch_first=False, norm_first=False, device=None, dtype=None) -> None:
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, batch_first, norm_first, device, dtype)
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.history_multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                 **factory_kwargs)
        self.history_dropout = Dropout(dropout)

    def forward(self, tgt: Tensor, memory: Tensor, history_out: Tensor = None) -> Tensor:
        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), None, None)
            if history_out is None:  # 如果history_out不为None，利用额外的mha进行cross attention
                x = x + self._mha_block(self.norm2(x), memory, None, None)
            else:
                x = x + self._mha_block(self.norm2(x), memory, None, None) + self._history_mha_block(self.norm2(x), history_out)

            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x))            
            if history_out is None:  # 如果history_out不为None，利用额外的mha进行cross attention
                x = self.norm2(x + self._mha_block(x, memory, None, None))
            else:
                x = self.norm2(x + self._mha_block(x, memory, None, None) + self._history_mha_block(x, history_out))

            x = self.norm3(x + self._ff_block(x))

        return x

    # multihead attention block for `history_out`
    def _history_mha_block(self, x: Tensor, history_out: Tensor) -> Tensor:
        x = self.history_multihead_attn(x, history_out, history_out,
                                attn_mask=None, key_padding_mask=None, need_weights=False)[0]
        return self.history_dropout(x)