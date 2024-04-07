from typing import Optional, Union, Tuple
from typing import Optional, Dict, Tuple, Union, Any
from functools import partial
from eqxvision.layers import (
    # get_normalization_layer,
    # LinearLayer,
    # get_activation_fn,
    ConvLayer,
    # MultiHeadAttention,
    # Dropout,
    # SingleHeadAttention,
    # LinearSelfAttention,
)

import equinox as eqx
import jax
from jax import numpy as jnp

def get_normalization_layer(
    opts,
    num_features: int,
    norm_type: Optional[str] = None,
    num_groups: Optional[int] = None,
    *args,
    **kwargs
):
    """
    Helper function to get normalization layers
    """

    norm_type = (
        getattr(opts, "model.normalization.name", "batch_norm")
        if norm_type is None
        else norm_type
    )
    num_groups = (
        getattr(opts, "model.normalization.groups", 1)
        if num_groups is None
        else num_groups
    )
    momentum = getattr(opts, "model.normalization.momentum", 0.1)

    norm_layer = None
    norm_type = norm_type.lower() if norm_type is not None else None
    if norm_type in ["batch_norm", "batch_norm_2d"]:
        norm_layer = eqx.nn.BatchNorm(num_features, momentum=momentum, axis_name='batch')
    elif norm_type in ["layer_norm", "ln"]:
        norm_layer = eqx.nn.LayerNorm(num_features)
    else:
        print(norm_type)
        assert False
    
    return norm_layer

def get_activation_fn(
    act_type: Optional[str] = "relu",
    num_parameters: Optional[int] = -1,
    inplace: Optional[bool] = True,
    negative_slope: Optional[float] = 0.1,
    *args,
    **kwargs
):
    """
    Helper function to get activation (or non-linear) function
    """

    if act_type == "relu":
        return eqx.nn.Lambda(jax.nn.relu)
    # elif act_type == "prelu":
    #     assert num_parameters >= 1
    #     return PReLU(num_parameters=num_parameters)
    # elif act_type == "leaky_relu":
    #     return LeakyReLU(negative_slope=negative_slope, inplace=inplace)
    # elif act_type == "hard_sigmoid":
    #     return Hardsigmoid(inplace=inplace)
    elif act_type == "swish":
        return eqx.nn.Lambda(jax.nn.swish)
    # elif act_type == "gelu":
    #     return GELU()
    # elif act_type == "sigmoid":
    #     return Sigmoid()
    # elif act_type == "relu6":
    #     return ReLU6(inplace=inplace)
    # elif act_type == "hard_swish":
    #     return Hardswish(inplace=inplace)
    # elif act_type == "tanh":
    #     return Tanh()
    else:
        print(
            "Supported activation layers are. Supplied argument is: {}".format(
                act_type
            )
        )
        assert False

#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

# import torch
# from torch import nn, Tensor
from typing import Optional, Tuple
# from torch.nn import functional as F

# from utils import logger

# from .base_layer import BaseLayer
# from .linear_layer import LinearLayer
# from .dropout import Dropout
# from ..misc.profiler import module_profile


class MultiHeadAttention(eqx.Module):
    qkv_proj: eqx.Module
    attn_dropout: eqx.Module
    out_proj: eqx.Module
    softmax: eqx.Module
    
    head_dim: Any = eqx.field(static=True)
    scaling: Any = eqx.field(static=True)
    num_heads: Any = eqx.field(static=True)
    embed_dim: Any = eqx.field(static=True)
    coreml_compatible: Any = eqx.field(static=True)
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attn_dropout: Optional[float] = 0.0,
        bias: Optional[bool] = True,
        coreml_compatible: Optional[bool] = False,
        key = None,
        *args,
        **kwargs
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            print(
                "Embedding dim must be divisible by number of heads in {}. Got: embed_dim={} and num_heads={}".format(
                    self.__class__.__name__, embed_dim, num_heads
                )
            )
            assert False

        keys = jax.random.split(key, 2)
        self.qkv_proj = eqx.nn.Linear(
            in_features=embed_dim, out_features=3 * embed_dim, use_bias=bias, key=keys[0],
        )

        self.attn_dropout = eqx.nn.Dropout(p=attn_dropout)
        self.out_proj = eqx.nn.Linear(
            in_features=embed_dim, out_features=embed_dim, use_bias=bias, key=keys[1]
        )

        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.softmax = eqx.nn.Lambda(partial(jax.nn.softmax, axis=-1))
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.coreml_compatible = coreml_compatible


    def forward_default(self, x_q, x_kv = None, **kwargs):

        # [N, P, C]
        # b_sz, n_patches, in_channels = x_q.shape

        if x_kv is None:
            # self-attention
            # [N, P, C] --> [N, P, 3C] --> [N, P, 3, h, c] where C = hc
            x_q = jax.vmap(self.qkv_proj)(x_q)
            qkv = jnp.reshape(x_q,(kwargs["b_sz"], kwargs["n_patches"], 3, self.num_heads, -1))
            # [N, P, 3, h, c] --> [N, h, 3, P, C]
            qkv = jnp.transpose(qkv, (0,3,2,1,4))

            # [N, h, 3, P, C] --> [N, h, P, C] x 3
            query, key, value = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        else:
            # cross-attention
            assert False


        query = query * self.scaling

        # [N h, P, c] --> [N, h, c, P]
        # key = key.transpose(-1, -2)
        key = jnp.transpose(key, (0,1,3,2))

        # QK^T
        # [N, h, P, c] x [N, h, c, P] --> [N, h, P, P]
        attn = jnp.matmul(query, key)
        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)

        # weighted sum
        # [N, h, P, P] x [N, h, P, c] --> [N, h, P, c]
        out = jnp.matmul(attn, value)

        # [N, h, P, c] --> [N, P, h, c] --> [N, P, C]
        # out = out.transpose(1, 2)
        out = jnp.transpose(out, (0,2,1,3))
        out = jnp.reshape(out, (kwargs["b_sz"]*kwargs["n_patches"], -1))
        out = jax.vmap(self.out_proj)(out)
        # out = jnp.reshape(out, (b_sz, n_patches, -1))

        return out

    def __call__(
        self, x_q, x_kv = None, *args, **kwargs
    ):
        if self.coreml_compatible:
            assert False
        else:
            return self.forward_default(x_q=x_q, x_kv=x_kv, **kwargs)


class TransformerEncoder(eqx.Module):
    pre_norm_mha: eqx.Module
    pre_norm_ffn: eqx.Module
    
    embed_dim: Any = eqx.field(static=True)
    ffn_dim: Any = eqx.field(static=True)
    ffn_dropout: Any = eqx.field(static=True)
    std_dropout: Any = eqx.field(static=True)
    attn_fn_name: Any = eqx.field(static=True)
    act_fn_name: Any = eqx.field(static=True)
    norm_type: Any = eqx.field(static=True)
    
    def __init__(
        self,
        opts,
        embed_dim: int,
        ffn_latent_dim: int,
        num_heads: Optional[int] = 8,
        attn_dropout: Optional[float] = 0.0,
        dropout: Optional[float] = 0.0,
        ffn_dropout: Optional[float] = 0.0,
        transformer_norm_layer: Optional[str] = "layer_norm",
        key = None,
        *args,
        **kwargs
    ) -> None:

        super().__init__()

        keys = jax.random.split(key, 3)
        if num_heads > 1:
            attn_unit = MultiHeadAttention(
                embed_dim,
                num_heads,
                attn_dropout=attn_dropout,
                bias=True,
                coreml_compatible=getattr(
                    opts, "common.enable_coreml_compatible_module", False
                ), key = keys[0],
                
            )
        else:
            assert False
            # attn_unit = SingleHeadAttention(
            # embed_dim=embed_dim, attn_dropout=attn_dropout, bias=True
            # )
        self.pre_norm_mha = eqx.nn.Sequential([
            get_normalization_layer(
                opts=opts, norm_type=transformer_norm_layer, num_features=embed_dim
            ),
            attn_unit,
            eqx.nn.Dropout(p=dropout),
        ]
        )

        act_name = self.build_act_layer(opts=opts)
        self.pre_norm_ffn = eqx.nn.Sequential([
            get_normalization_layer(
                opts=opts, norm_type=transformer_norm_layer, num_features=embed_dim
            ),
            eqx.nn.Linear(in_features=embed_dim, out_features=ffn_latent_dim, use_bias=True, key=keys[1]),
            act_name,
            eqx.nn.Dropout(p=ffn_dropout),
            eqx.nn.Linear(in_features=ffn_latent_dim, out_features=embed_dim, use_bias=True, key=keys[2]),
            eqx.nn.Dropout(p=dropout),
        ]
        )
        
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_latent_dim
        self.ffn_dropout = ffn_dropout
        self.std_dropout = dropout
        self.attn_fn_name = attn_unit.__class__.__name__
        self.act_fn_name = act_name.__class__.__name__
        self.norm_type = transformer_norm_layer

    @staticmethod
    def build_act_layer(opts):
        act_type = getattr(opts, "model.activation.name", "relu")
        neg_slope = getattr(opts, "model.activation.neg_slope", 0.1)
        inplace = getattr(opts, "model.activation.inplace", False)
        act_layer = get_activation_fn(
            act_type=act_type,
            inplace=inplace,
            negative_slope=neg_slope,
            num_parameters=1,
        )
        return act_layer

    def __call__(
        self, x, x_prev = None, key=None, *args, **kwargs
    ):

        # Multi-head attention
        keys = jax.random.split(key, 2)
        
        
        # print(x.shape)
        
        b_sz, n_patches, in_channels = x.shape
        x = jnp.reshape(x, (b_sz*n_patches, in_channels))
        res = x
        
        x = jax.vmap(self.pre_norm_mha[0])(x)  # norm
        # x = jnp.reshape(x, shape_old)
        
        x = self.pre_norm_mha[1](x_q=x, x_kv=x_prev, b_sz=b_sz, n_patches=n_patches, in_channels=in_channels)  # mha
        
        x = self.pre_norm_mha[2](x, key=keys[0])  # dropout
        x = x + res

        # Feed forward network
        keys = jax.random.split(keys[1], b_sz*n_patches)
        x = x + jax.vmap(lambda x, key: self.pre_norm_ffn(x, key=key), in_axes=(0,0))(x, keys)
        x = jnp.reshape(x, (b_sz, n_patches, -1))
        return x
