import copy
from typing import Optional, Any, Union, Callable
import math

import torch
from torch import nn
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


class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)
    
class TemporalTransformer(Module):
    def __init__(self, opt, batch_first: bool = False, d_model = 768, nhead: int = 8, num_encoder_layers: int = 6, num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1) -> None:
        super(TemporalTransformer, self).__init__()
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu
        layer_norm_eps: float = 1e-5
        norm_first: bool = False
        device=None
        dtype=None
        factory_kwargs = {'device': device, 'dtype': dtype}

        # encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
        #                                         activation, layer_norm_eps, batch_first, norm_first,
        #                                         **factory_kwargs)
        # self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        encoder_layer = TemporalTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                 activation, layer_norm_eps, batch_first, norm_first,
                                                 **factory_kwargs)
        encoder_norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        decoder_layer = TemporalTransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                activation, layer_norm_eps, batch_first, norm_first, opt.cross_first,
                                                **factory_kwargs)
        decoder_norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        if opt.per_pe:
            self.encoder = TemporalTransformerEncoderWithPE(d_model, encoder_layer, opt.encoder_layer_num, encoder_norm)
            self.decoder = TemporalTransformerDecoderWithPE(d_model, decoder_layer, opt.decoder_layer_num, decoder_norm)
        else:
            self.encoder = TemporalTransformerEncoder(encoder_layer, opt.encoder_layer_num, encoder_norm)
            self.decoder = TemporalTransformerDecoder(decoder_layer, opt.decoder_layer_num, decoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

        self.batch_first = batch_first

    def forward(self, src: Tensor, tgt: Tensor, history_out = None, tgt_mask: Optional[Tensor] = None) -> Tensor:

        if not self.batch_first and src.size(1) != tgt.size(1):
            raise RuntimeError("the batch number of src and tgt must be equal")
        elif self.batch_first and src.size(0) != tgt.size(0):
            raise RuntimeError("the batch number of src and tgt must be equal")

        if src.size(2) != self.d_model or tgt.size(2) != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")

        memory = self.encoder(src, None, None)
        output = self.decoder(tgt, memory, history_out = history_out, tgt_mask=tgt_mask)

        return output

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

class TemporalTransformerEncoderLayer(TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu, layer_norm_eps=1e-5, batch_first=False, norm_first=False, device=None, dtype=None) -> None:
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, batch_first, norm_first, device, dtype)
        factory_kwargs = {'device': device, 'dtype': dtype}
    
    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        x = src

        x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
        x = self.norm2(x + self._ff_block(x))

        return x

class TemporalTransformerEncoder(TransformerEncoder):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__(encoder_layer, num_layers, norm)
        
    def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        output = src

        for i, mod in enumerate(self.layers):
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)
        return output

# Transformer with positional encoding in each layer
class TemporalTransformerEncoderWithPE(TransformerEncoder):
    def __init__(self, latent_dim, encoder_layer, num_layers, norm=None):
        super().__init__(encoder_layer, num_layers, norm)
        self.positional_encoding = PositionalEncoding(emb_size=latent_dim, dropout=0.1)
        
    def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        output = src

        for i, mod in enumerate(self.layers):
            output = self.positional_encoding(output.transpose(1, 0)).transpose(1, 0)  
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)
        return output

class TemporalTransformerDecoderWithPE(TransformerDecoder):
    def __init__(self, latent_dim, decoder_layer, num_layers, norm=None):
        super().__init__(decoder_layer, num_layers, norm)
        self.positional_encoding = PositionalEncoding(emb_size=latent_dim, dropout=0.1)
        
    def forward(self, tgt: Tensor, memory: Tensor, history_out: Tensor = None, tgt_mask: Optional[Tensor] = None) -> Tensor:
        output = tgt

        for i, mod in enumerate(self.layers):
            output = self.positional_encoding(output.transpose(1, 0)).transpose(1, 0)
            output = mod(output, memory, history_out, tgt_mask)  # 传入history_out

        if self.norm is not None:
            output = self.norm(output)

        return output

class TemporalTransformerDecoder(TransformerDecoder):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__(decoder_layer, num_layers, norm)
    
    def forward(self, tgt: Tensor, memory: Tensor, history_out: Tensor = None, tgt_mask: Optional[Tensor] = None) -> Tensor:
        output = tgt

        for mod in self.layers:
            output = mod(output, memory, history_out, tgt_mask)  # 传入history_out

        if self.norm is not None:
            output = self.norm(output)

        return output


class TemporalTransformerDecoderLayer(TransformerDecoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu, layer_norm_eps=0.00001, batch_first=False, norm_first=False, 
        cross_first=False, device=None, dtype=None) -> None:
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, batch_first, norm_first, device, dtype)
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.history_multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                 **factory_kwargs)
        self.history_dropout = Dropout(dropout)

        self.cross_first = cross_first

    def forward(self, tgt: Tensor, memory: Tensor, history_out: Tensor = None, tgt_mask: Optional[Tensor] = None) -> Tensor:
        x = tgt
        if self.cross_first is True:
            x = self.norm2(x + self._mha_block(x, memory, None, None))
            x = self.norm1(x + self._sa_block(x, attn_mask=tgt_mask, key_padding_mask=None))            
        else:    
            x = self.norm1(x + self._sa_block(x, attn_mask=tgt_mask, key_padding_mask=None))            
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