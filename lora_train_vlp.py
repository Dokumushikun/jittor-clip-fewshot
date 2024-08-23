import os
import random
import numpy as np
import argparse

from jittor.nn import Module, Linear, softmax, pad, linear, dropout

from PIL import Image
import os.path as osp
import tarfile
import zipfile
import gdown
from collections import defaultdict
import math
from tqdm import tqdm
import jittor as jt
from jittor import nn, Module
from jittor import transform as T
import jittor.nn as F
# import clip
from jclip import clip
from jittor import attention, misc, dataset
from jittor.dataset import Dataset, DataLoader

jt.flags.use_cuda = 1

INDEX_POSITIONS_TEXT = {
    'top1': [11],
    'top2': [10, 11],
    'top3': [9, 10, 11],
    'bottom': [0, 1, 2, 3],
    'mid': [4, 5, 6, 7],
    'up': [8, 9, 10, 11],
    'half-up': [6, 7, 8, 9, 10, 11],
    'half-bottom': [0, 1, 2, 3, 4, 5],
    'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}

INDEX_POSITIONS_VISION = {
    'ViT-B/16': {
        'top': [11],
        'top3': [9, 10, 11],
        'bottom': [0, 1, 2, 3],
        'mid': [4, 5, 6, 7],
        'up': [8, 9, 10, 11],
        'half-up': [6, 7, 8, 9, 10, 11],
        'half-bottom': [0, 1, 2, 3, 4, 5],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]},
    'ViT-B/32': {
        'bottom': [0, 1, 2, 3],
        'mid': [4, 5, 6, 7],
        'up': [8, 9, 10, 11],
        'half-up': [6, 7, 8, 9, 10, 11],
        'half-bottom': [0, 1, 2, 3, 4, 5],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]},

    'ViT-L/14': {
        'bottom': [0, 1, 2, 3],
        'mid': [4, 5, 6, 7],
        'up': [8, 9, 10, 11],
        'half-up': [6, 7, 8, 9, 10, 11],
        'half-bottom': [0, 1, 2, 3, 4, 5],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]}
}


def set_param(curr_mod, name, param=None, mode='update'):
    r"""Refer to https://github.com/Baijiong-Lin/MOML/blob/main/MTL/utils.py"""
    if '.' in name:
        n = name.split('.')
        module_name = n[0]
        rest = '.'.join(n[1:])
        for name, mod in curr_mod.named_children():
            if module_name == name:
                return set_param(mod, rest, param, mode=mode)
    else:
        if mode == 'update':
            delattr(curr_mod, name)
            setattr(curr_mod, name, param)
        elif mode == 'get':
            if hasattr(curr_mod, name):
                p = getattr(curr_mod, name)
                return p


import glob
import os


def load_class_names(filepath):
    file_paths = glob.glob(os.path.join(filepath, "*.txt"))
    prompts_dict = {}
    for file_path in file_paths:
        with open(file_path, "r") as file:
            for i, line in enumerate(file):
                line_content = line.strip()
                if i in prompts_dict:
                    prompts_dict[i].append(line_content)
                else:
                    prompts_dict[i] = [line_content]

    return prompts_dict

def load_class_names_random(filepath):
    filename = 'text_template' + str(random.randrange(1, 9)) + '.txt'
    #filename2 = 'text_template' + str(random.randrange(5, 9)) + '.txt'
    file_paths = [os.path.join(filepath, filename)]
    prompts_dict = {}
    for file_path in file_paths:
        with open(file_path, "r") as file:
            for i, line in enumerate(file):
                line_content = line.strip()
                if i in prompts_dict:
                    prompts_dict[i].append(line_content)
                else:
                    prompts_dict[i] = [line_content]

    return prompts_dict
    
template = load_class_names('text_template')


def get_lora_parameters(model, bias='none'):
    params = []
    for name, param in model.named_parameters():
        if bias == 'none':
            if 'lora_' in name:
                params.append(param)
        elif bias == 'all':
            if 'lora_' in name or 'bias' in name:
                params.append(param)
        elif bias == 'lora_only':
            if 'lora_' in name:
                params.append(param)
                bias_name = name.split('lora_')[0] + 'bias'
                if bias_name in model.state_dict():
                    bias_param = dict(model.named_parameters())[bias_name]
                    params.append(bias_param)
        else:
            raise NotImplementedError
    return params


def mark_only_lora_as_trainable(model: Module, bias: str = 'none') -> None:
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad_ = False
    if bias == 'none':
        return
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad_ = True
    elif bias == 'lora_only':
        for m in model.modules():
            if isinstance(m, LoRALayer) and \
                    hasattr(m, 'bias') and \
                    m.bias is not None:
                m.bias.requires_grad_ = True
    else:
        raise NotImplementedError


def lora_state_dict(model: Module, bias: str = 'none'):
    my_state_dict = model.state_dict()
    if bias == 'none':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k}
    elif bias == 'all':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k or 'bias' in k}
    elif bias == 'lora_only':
        to_return = {}
        for k in my_state_dict:
            if 'lora_' in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split('lora_')[0] + 'bias'
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError


import warnings


class LoRALayer():
    def __init__(
            self,
            r: int,
            lora_alpha: int,
            fan_in_fan_out: bool = False,
            dropout_rate: float = 0,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        self.dropout_rate = dropout_rate
        if self.r > 0:
            self.scaling = self.lora_alpha / math.sqrt(self.r)
        self.merged = False
        self.fan_in_fan_out = fan_in_fan_out
        self.params_with_lora = {}

    def register_lora_param(self):
        for param_name, lora_name in self.params_with_lora.items():
            assert len(eval(f'self.{param_name}').shape) == 2
            self.__dict__[f'{lora_name}_lora_A'] = jt.zeros((self.r, eval(f'self.{param_name}').shape[1]))
            self.__dict__[f'{lora_name}_lora_B'] = jt.zeros((eval(f'self.{param_name}').shape[0], self.r))
            eval(f'self.{param_name}').requires_grad = False

    def init_lora_param(self):
        for param_name, lora_name in self.params_with_lora.items():
            if hasattr(self, f'{lora_name}_lora_A'):
                nn.init.kaiming_uniform_(eval(f'self.{lora_name}_lora_A'), a=math.sqrt(5))
                nn.init.zero_(eval(f'self.{lora_name}_lora_B'))

    def transpose(self, w: jt.Var):
        return w.transpose(0, 1) if self.fan_in_fan_out else w

    def merge_BA(self, param_name: str):
        lora_name = self.params_with_lora[param_name]
        return self.transpose((eval(f'self.{lora_name}_lora_B') @ eval(f'self.{lora_name}_lora_A')).reshape(
            eval(f'self.{param_name}').shape))

    def merge_lora_param(self):
        for param_name, lora_name in self.params_with_lora.items():
            p = self.__dict__[param_name]
            p_new = p.detach() + self.merge_BA(param_name) * self.scaling
            self.__dict__[param_name] = p_new

    def add_lora_data(self):
        for param_name, lora_name in self.params_with_lora.items():
            eval(f'self.{param_name}').data += self.merge_BA(param_name) * self.scaling

    def sub_lora_data(self):
        for param_name, lora_name in self.params_with_lora.items():
            eval(f'self.{param_name}').data -= self.merge_BA(param_name) * self.scaling

    def lora_train(self, mode: bool = True):
        if mode:
            if self.merged and self.r > 0:
                self.sub_lora_data()
            self.merged = False
        else:
            if not self.merged and self.r > 0:
                self.add_lora_data()
            self.merged = True


class LinearLoRA(nn.Linear, LoRALayer):
    def __init__(
            self,
            existing_linear: nn.Linear,
            r: int = 0,
            lora_alpha: int = 1,
            fan_in_fan_out: bool = False,
            dropout_rate=0.,
            **kwargs
    ):
        super().__init__(
            in_features=existing_linear.in_features,
            out_features=existing_linear.out_features)

        self.copy_parameters_from_existing(existing_linear)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, fan_in_fan_out=fan_in_fan_out)

        self.params_with_lora = {'weight': 'w'}
        if r > 0:
            self.register_lora_param()
        self.init_lora_param()
        self.weight.data = self.transpose(self.weight.data)
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

    def copy_parameters_from_existing(self, existing_linear):
        self.weight.data = jt.array(existing_linear.weight.data)
        if existing_linear.bias is not None:
            self.bias = jt.array(existing_linear.bias.data)
        else:
            self.bias = None

    def train(self, mode: bool = True):
        super().train(mode)
        self.lora_train(mode)

    def execute(self, x: jt.Var, **kwargs):
        if self.dropout is None:
            if self.r > 0 and not self.merged:
                self.merge_lora_param()
                result = super().execute(x, **kwargs)
                self.sub_lora_data()
                return result
            else:
                return super().execute(x, **kwargs)

        original_output = super().execute(x)

        if self.is_training() and self.dropout.p > 0:
            x = self.dropout(x)

        if self.r > 0 and not self.merged:
            lora_adjustment = (x @ self.merge_BA('weight').transpose(0, 1)) * self.scaling
            result = original_output + lora_adjustment
        else:
            result = original_output
        return result


from typing import Optional, Tuple


def _canonical_mask(
        mask,
        mask_name: str,
        # other_type,
        other_name: str,
        target_type,
        check_other: bool = True,
):
    if mask is not None:
        _mask_dtype = mask.dtype
        _mask_is_float = mask.dtype == jt.float16 or mask.dtype == jt.float32 or mask.dtype == jt.float64
        if _mask_dtype != jt.bool and not _mask_is_float:
            raise AssertionError(
                f"only bool and floating types of {mask_name} are supported")
        '''            if check_other and other_type is not None:
            if _mask_dtype != other_type:
                warnings.warn(
                    f"Support for mismatched {mask_name} and {other_name} "
                    "is deprecated. Use same type for both instead.")'''
        if not _mask_is_float:
            # WARNING(514flowey): Check Here
            new_mask = jt.zeros_like(mask, dtype=target_type)
            new_mask[mask] = float("-inf")
            mask = new_mask
    return mask


def scaled_dot_product_attention(query,
                                 key,
                                 value,
                                 attn_mask=None,
                                 dropout_p=0.0,
                                 is_causal=False,
                                 scale=None,
                                 training=True) -> jt.Var:
    # Efficient implementation equivalent to the following:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = jt.zeros(L, S, dtype=query.dtype)
    if is_causal:
        assert attn_mask is None
        temp_mask = jt.ones(L, S, dtype=jt.bool).tril(diagonal=0)
        attn_bias[jt.logical_not(temp_mask)] = float("-inf")
        # attn_bias.to(query.dtype)
        attn_bias = jt.array(attn_bias, query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == jt.bool:
            attn_bias[jt.logical_not(temp_mask)] = float("-inf")
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = softmax(attn_weight, dim=-1)
    attn_weight = dropout(attn_weight, dropout_p, is_train=training)
    return attn_weight @ value


def _none_or_dtype(input):
    if input is None:
        return None
    elif isinstance(input, jt.Var):
        return input.dtype


class PlainMultiheadAttentionLoRA(nn.Module):
    def __init__(self, existing_mha: attention.MultiheadAttention, enable_lora: list = ['q', 'k', 'v', 'o'], r: int = 0,
                 lora_alpha: int = 1, dropout_rate: float = 0., **kwargs):
        super().__init__()

        self.dropout = 0
        self.embed_dim = existing_mha.embed_dim
        self.kdim = existing_mha.kdim
        self.vdim = existing_mha.vdim
        self._qkv_same_embed_dim = existing_mha._qkv_same_embed_dim
        self.num_heads = existing_mha.num_heads
        self.batch_first = existing_mha.batch_first
        self.head_dim = existing_mha.head_dim
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=existing_mha.in_proj_bias is not None)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=existing_mha.in_proj_bias is not None)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=existing_mha.in_proj_bias is not None)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim, bias=existing_mha.out_proj.bias is not None)

        with jt.no_grad():
            existing_weight = jt.array(existing_mha.in_proj_weight.data)
            existing_bias = jt.array(existing_mha.in_proj_bias.data) if existing_mha.in_proj_bias is not None else None

            self.q_proj.weight.data = existing_weight[:self.embed_dim, :]
            if existing_bias is not None:
                self.q_proj.bias.data = existing_bias[:self.embed_dim]

            self.k_proj.weight.data = existing_weight[self.embed_dim:2 * self.embed_dim, :]
            if existing_bias is not None:
                self.k_proj.bias.data = existing_bias[self.embed_dim:2 * self.embed_dim]

            self.v_proj.weight.data = existing_weight[2 * self.embed_dim:, :]
            if existing_bias is not None:
                self.v_proj.bias.data = existing_bias[2 * self.embed_dim:]

            self.proj.weight.data = jt.array(existing_mha.out_proj.weight.data)
            if self.proj.bias is not None:
                self.proj.bias.data = jt.array(existing_mha.out_proj.bias.data)

        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, dropout_rate=dropout_rate)

        for item in enable_lora:
            if item == 'q':
                self.q_proj = LinearLoRA(self.q_proj, r=r, lora_alpha=lora_alpha, fan_in_fan_out=False,
                                         dropout_rate=dropout_rate)
            elif item == 'k':
                self.k_proj = LinearLoRA(self.k_proj, r=r, lora_alpha=lora_alpha, fan_in_fan_out=False,
                                         dropout_rate=dropout_rate)
            elif item == 'v':
                self.v_proj = LinearLoRA(self.v_proj, r=r, lora_alpha=lora_alpha, fan_in_fan_out=False,
                                         dropout_rate=dropout_rate)
            elif item == 'o':
                self.proj = LinearLoRA(self.proj, r=r, lora_alpha=lora_alpha, fan_in_fan_out=False,
                                       dropout_rate=dropout_rate)

    def forward_module(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None,
                       average_attn_weights=True, is_causal=False):
        if attn_mask is not None and is_causal:
            raise AssertionError("Only allow causal mask or attn_mask")
        is_batched = query.ndim == 3
        key_padding_mask = _canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            # other_type=_none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype
        )
        attn_mask = _canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            # other_type=_none_or_dtype(key_padding_mask),
            other_name="key_padding_mask",
            target_type=query,
            check_other=False,
        )

        if self.batch_first and is_batched:
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = [x.transpose(1, 0) for x in (query, key)]
                    value = key
            else:
                query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        if attn_mask is not None:
            if attn_mask.ndim == 2:
                correct_2d_size = (tgt_len, src_len)
                if attn_mask.shape != correct_2d_size:
                    raise RuntimeError(
                        f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.ndim == 3:
                correct_3d_size = (bsz * self.num_heads, tgt_len, src_len)
                if attn_mask.shape != correct_3d_size:
                    raise RuntimeError(
                        f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
            else:
                raise RuntimeError(f"attn_mask's dimension {attn_mask.ndim} is not supported")

        if attn_mask is not None:
            if attn_mask.shape[0] == 1 and attn_mask.ndim == 3:
                attn_mask = attn_mask.unsqueeze(0)
            else:
                attn_mask = attn_mask.view(bsz, self.num_heads, -1, src_len)

        dropout_p = self.dropout if self.is_training() else 0.

        q = q.view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        src_len = k.shape[1]
        q = q.view(bsz, self.num_heads, tgt_len, self.head_dim)
        k = k.view(bsz, self.num_heads, src_len, self.head_dim)
        v = v.view(bsz, self.num_heads, src_len, self.head_dim)

        attn_output = scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal)
        attn_output = attn_output.permute(2, 0, 1, 3).reshape(bsz * tgt_len, embed_dim)
        attn_output = self.proj(attn_output)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.shape[1])
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), None
        return attn_output, None

    def train(self, mode: bool = True):
        super().train(mode)
        self.lora_train(mode)

    def execute(self, query: jt.Var, key: jt.Var, value: jt.Var, **kwargs):
        return self.forward_module(query, key, value, **kwargs)


def apply_lora(args, clip_model):
    list_lora_layers = []

    if args.encoder == 'text' or args.encoder == 'both':
        indices = INDEX_POSITIONS_TEXT[args.position]
        text_encoder = clip_model.transformer
        for i, block in enumerate(text_encoder.resblocks):
            # print(f"Residual Attention Block {i}: {block}")
            if i in indices:
                submodule = block.attn
                if submodule.__class__.__name__ == 'MultiheadAttention':
                    print(f"Submodule at block {i} is {submodule.__class__.__name__}")
                    new_multi_head_lora = PlainMultiheadAttentionLoRA(
                        submodule, enable_lora=args.params, r=args.r, lora_alpha=args.alpha,
                        dropout_rate=args.dropout_rate)
                    block.attn = new_multi_head_lora
                    list_lora_layers.append(new_multi_head_lora)

    if args.encoder == 'vision' or args.encoder == 'both':
        indices = INDEX_POSITIONS_VISION[args.backbone][args.position]
        vision_encoder = clip_model.visual.transformer
        for i, block in enumerate(vision_encoder.resblocks):
            # print(f"Residual Attention Block {i}: {block}")
            if i in indices:
                submodule = block.attn
                if submodule.__class__.__name__ == 'MultiheadAttention':
                    new_multi_head_lora = PlainMultiheadAttentionLoRA(
                        submodule, enable_lora=args.params, r=args.r, lora_alpha=args.alpha,
                        dropout_rate=args.dropout_rate)
                    block.attn = new_multi_head_lora
                    list_lora_layers.append(new_multi_head_lora)

    return list_lora_layers


def save_lora(args, epoch, list_lora_layers):
    weights = {}
    for i, layer in enumerate(list_lora_layers):
        layer_weights = {}
        if 'q' in args.params:
            layer_weights['q_proj'] = {
                'w_lora_A': layer.q_proj.w_lora_A.data,
                'w_lora_B': layer.q_proj.w_lora_B.data
            }
        if 'k' in args.params:
            layer_weights['k_proj'] = {
                'w_lora_A': layer.k_proj.w_lora_A.data,
                'w_lora_B': layer.k_proj.w_lora_B.data
            }
        if 'v' in args.params:
            layer_weights['v_proj'] = {
                'w_lora_A': layer.v_proj.w_lora_A.data,
                'w_lora_B': layer.v_proj.w_lora_B.data
            }
        if 'o' in args.params:
            layer_weights['proj'] = {
                'w_lora_A': layer.proj.w_lora_A.data,
                'w_lora_B': layer.proj.w_lora_B.data
            }

        weights[f'layer_{i}'] = layer_weights

    metadata = {
        'r': args.r,
        'alpha': args.alpha,
        'encoder': args.encoder,
        'params': args.params,
        'position': args.position
    }

    save_data = {
        'weights': weights,
        'metadata': metadata
    }

    save_path = f'lora_weights1/lora_weights.pkl'
    jt.save(save_data, save_path)
    print(f'LoRA weights saved to {save_path}')


def load_lora(args, list_lora_layers, load_path):
    if not os.path.exists(load_path):
        raise FileNotFoundError(f'File {load_path} does not exist.')

    loaded_data = jt.load(load_path)

    metadata = loaded_data['metadata']
    if metadata['r'] != args.r:
        raise ValueError(
            f"r mismatch: expected {args.r}, found {metadata['r']}")
    if metadata['alpha'] != args.alpha:
        raise ValueError(
            f"alpha mismatch: expected {args.alpha}, found {metadata['alpha']}")
    if metadata['encoder'] != args.encoder:
        raise ValueError(
            f"Encoder mismatch: expected {args.encoder}, found {metadata['encoder']}")
    if metadata['params'] != args.params:
        raise ValueError(
            f"Params mismatch: expected {args.params}, found {metadata['params']}")
    if metadata['position'] != args.position:
        raise ValueError(
            f"Position mismatch: expected {args.position}, found {metadata['position']}")

    weights = loaded_data['weights']
    for i, layer in enumerate(list_lora_layers):
        layer_weights = weights[f'layer_{i}']
        if 'q' in args.params and 'q_proj' in layer_weights:
            layer.q_proj.w_lora_A.data = jt.array(layer_weights['q_proj']['w_lora_A'])
            layer.q_proj.w_lora_B.data = jt.array(layer_weights['q_proj']['w_lora_B'])
        if 'k' in args.params and 'k_proj' in layer_weights:
            layer.k_proj.w_lora_A.data = jt.array(layer_weights['k_proj']['w_lora_A'])
            layer.k_proj.w_lora_B.data = jt.array(layer_weights['k_proj']['w_lora_B'])
        if 'v' in args.params and 'v_proj' in layer_weights:
            layer.v_proj.w_lora_A.data = jt.array(layer_weights['v_proj']['w_lora_A'])
            layer.v_proj.w_lora_B.data = jt.array(layer_weights['v_proj']['w_lora_B'])
        if 'o' in args.params and 'proj' in layer_weights:
            layer.proj.w_lora_A.data = jt.array(layer_weights['proj']['w_lora_A'])
            layer.proj.w_lora_B.data = jt.array(layer_weights['proj']['w_lora_B'])

    print(f'LoRA weights loaded from {load_path}')


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.equal(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).numpy())
    acc = 100 * acc / target.shape[0]

    return acc


def clip_classifier(templates_dict, clip_model):
    clip_weights = []
    for category_id, templates in templates_dict.items():
        # Tokenize and process each list of templates for the current category
        class_embeddings = []
        for template in templates:
            texts = clip.tokenize([template])
            embedding = clip_model.encode_text(texts)
            embedding /= embedding.norm(dim=-1, keepdim=True)
            class_embeddings.append(embedding)

        # Average the embeddings for all templates in the current category
        class_embeddings = jt.stack(class_embeddings).mean(dim=0)
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        clip_weights.append(class_embeddings)

    # Stack all class embeddings into a weight matrix
    clip_weights = jt.stack(clip_weights, dim=1)

    return clip_weights


def set_random_seed(seed):
    # random.seed(seed)
    # np.random.seed(seed)
    misc.set_global_seed(seed)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=1, type=int)
    # Dataset arguments
    parser.add_argument('--root_path', type=str, default='datasets')
    parser.add_argument('--dataset', type=str, default='jt')
    parser.add_argument('--shots', default=4, type=int)
    # Model arguments
    parser.add_argument('--backbone', default='ViT-B/32', type=str)
    # Training arguments
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--n_iters', default=500, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    # LoRA arguments
    parser.add_argument('--position', type=str, default='all',
                        choices=['bottom', 'mid', 'up', 'half-up', 'half-bottom', 'all', 'top3'],
                        help='where to put the LoRA modules')
    parser.add_argument('--encoder', type=str, choices=['text', 'vision', 'both'], default='both')
    parser.add_argument('--params', metavar='N', type=str, nargs='+', default=['q', 'k', 'v'],
                        help='list of attention matrices where putting a LoRA')
    parser.add_argument('--r', default=4, type=int, help='the rank of the low-rank matrices')
    parser.add_argument('--alpha', default=1, type=int, help='scaling (see LoRA paper)')
    parser.add_argument('--dropout_rate', default=0.25, type=float, help='dropout rate applied before the LoRA module')

    parser.add_argument('--save_path', default=None,
                        help='path to save the lora modules after training, not saved if None')
    parser.add_argument('--filename', default='lora_weights',
                        help='file name to save the lora weights (.pt extension will be added)')

    parser.add_argument('--eval_only', default=False, action='store_true',
                        help='only evaluate the LoRA modules (save_path should not be None)')
    args = parser.parse_args()

    return args


def read_image(path):
    """Read image from path using ``PIL.Image``.

    Args:
        path (str): path to an image.

    Returns:
        PIL image
    """
    if not osp.exists(path):
        raise IOError('No file exists at {}'.format(path))

    while True:
        try:
            img = Image.open(path).convert('RGB')
            return img
        except IOError:
            print(
                'Cannot read image from {}, '
                'probably due to heavy IO. Will re-try'.format(path)
            )

def gaussian_kernel(mu, bandwidth, datapoints):
    dist = jt.norm(datapoints - mu, dim=-1, p=2)
    density = jt.exp(-dist**2/(2*bandwidth**2))
    return density
def cdist(x1, x2):
    x1_square = jt.sum(x1 ** 2, dim=1, keepdims=True)
    x2_square = jt.sum(x2 ** 2, dim=1, keepdims=True)
    dist = jt.sqrt(x1_square - 2 * jt.matmul(x1, x2.transpose()) + x2_square.transpose())
    return dist
def solve_mta(image_features,text_features):

    logits = image_features @ text_features * 100 
    
    lambda_y = 0.2
    lambda_q = 4
    max_iter = 5
    temperature = 1
    
    batch_size = image_features.shape[0]
    
    # bandwidth
    dist = cdist(image_features, image_features)
    _, sorted_dist = jt.argsort(dist, dim=1)
    k = int(0.3 * (image_features.shape[0]-1))
    selected_distances = sorted_dist[:, 1:k+1]**2  # exclude the distance to the point itself 
    mean_distance = jt.mean(selected_distances, dim=1)
    bandwidth = jt.sqrt(0.5 * mean_distance) 
    
    # Affinity matrix based on logits
    affinity_matrix = (logits/temperature).softmax(1) @ (logits/temperature).softmax(1).t()
    
    # Inlierness scores initialization: uniform
    y = jt.ones(batch_size) / batch_size
    
    # Mode initialization: original image embedding
    mode_init = image_features[0]
    mode = mode_init

    convergence = False
    th = 1e-6
    iter = 0
    
    while not convergence:
        # Inlierness step
        density = gaussian_kernel(mode, bandwidth, image_features)
    
        convergence_inlierness = False
        i = 0
        while not convergence_inlierness:
            i += 1
            old_y = y
            weighted_affinity = affinity_matrix * y.unsqueeze(0)
            y = nn.softmax(1/lambda_y * (density + lambda_q * jt.sum(weighted_affinity, dim=1)), dim=-1)

            if jt.norm(old_y - y) < th or i >= max_iter:

                convergence_inlierness = True
        
        # Mode step
        convergence_mode = False
        i = 0
        while not convergence_mode:

            i += 1
            old_mode = mode
            density = gaussian_kernel(mode, bandwidth, image_features)
            weighted_density = density * y
            mode = jt.sum(weighted_density.unsqueeze(1) * image_features, dim=0) / jt.sum(weighted_density)
            mode /= mode.norm(p=2, dim=-1)
            
            if jt.norm(old_mode - mode) < th or i >= max_iter:
                convergence_mode = True
        
        iter += 1
        if iter >= max_iter:
            convergence = True

    output = mode.unsqueeze(0) @ text_features * 100
    return output

def evaluate_lora(args, clip_model, loader):
    clip_model.eval()
    with jt.no_grad():
        template = load_class_names('text_template')
        textual_features = clip_classifier(template, clip_model).squeeze(0).t()
        acc = 0.
        acc1 = 0.
        acc2 = 0.
        tot_samples = 0
        for i, (image, images, target, impath) in enumerate(tqdm(loader)):
            image = jt.array(image)
            images = jt.array(images)
            image = image.squeeze(1)
            images = images.squeeze(1)
            #transformed_imgs = jt.stack(transformed_imgs)
            transformed_imgs = jt.concat((image, images))
            #print(transformed_imgs.shape)
            #transformed_imgs = jt.stack(transformed_imgs)
            #print(images.shape)
            image_features = clip_model.encode_image(transformed_imgs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            #cosine_similarity = image_features @ textual_features
            cosine_similarity_mta = solve_mta(image_features, textual_features)
            cosine_similarity_base = image_features[0].unsqueeze(0) @ textual_features
            #print(image_features.shape, textual_features.shape)
            cosine_similarity_ensemble = (image_features @ textual_features).mean(dim=0).unsqueeze(0)
            acc += cls_acc(cosine_similarity_mta, target) * len(cosine_similarity_mta)
            acc1 += cls_acc(cosine_similarity_base, target) * len(cosine_similarity_mta)
            acc2 += cls_acc(cosine_similarity_ensemble, target) * len(cosine_similarity_mta)
            tot_samples += len(cosine_similarity_mta)
    acc /= tot_samples
    acc1 /= tot_samples
    acc2 /= tot_samples
    return acc,acc1,acc2
    








def pre_load_features(clip_model, loader):
    features, labels = [], []
    with jt.no_grad():
        for i, (images, target,_) in enumerate(tqdm(loader)):
            images, target = images, target
            image_features = clip_model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            features.append(image_features)
            labels.append(target)
        features, labels = jt.concat(features), jt.concat(labels)

    return features, labels


class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = jt.array(param.data).clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if name not in self.shadow:
                    continue
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if name not in self.shadow:
                    continue
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if name not in self.shadow:
                    continue
                param.data = self.backup[name]
        self.backup = {}


def encode_text_in_batches(clip_model, texts, batch_size=32):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        if len(batch_texts) == 0:
            continue  # 跳过空批次
        batch_embeddings = clip_model.encode_text(clip.tokenize(batch_texts))
        all_embeddings.append(batch_embeddings)

    if len(all_embeddings) == 0:
        raise ValueError("No embeddings were generated. Please check your batch processing.")

    return jt.concat(all_embeddings, dim=0)

def run_lora(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader, val_loadercar):
    # Textual features
    print("\nGetting textual features as CLIP's classifier.")
    template = load_class_names('text_template')
    textual_features = clip_classifier(template, clip_model)
    textual_features = textual_features.squeeze(0).t()
    print(textual_features.size(0), textual_features.size(1))
    # Pre-load val features
    # print("\nLoading visual features and labels from val set.")
    # val_features, val_labels = pre_load_features(clip_model, val_loader)
    '''
    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(clip_model, val_loader)

    test_featurescar, test_labelscar = pre_load_features(clip_model, val_loadercar)
    # print(len(test_labels),len(test_labelscar))
    # Zero-shot CLIP
    clip_logits = logit_scale * test_features @ textual_features
    zs_acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(zs_acc))
    
    clip_logits = logit_scale * test_featurescar @ textual_features
    zs_acc = cls_acc(clip_logits, test_labelscar)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(zs_acc))
    '''
    list_lora_layers = apply_lora(args, clip_model)

    # ema = EMA(clip_model, 0.999)
    # ema.register()

    clip_model = clip_model.cuda()

    mark_only_lora_as_trainable(clip_model)
    total_epoch = 50

    clip_parameters = get_lora_parameters(clip_model)
    print(len(clip_parameters))
    all_parameters = list(clip_parameters)

    optimizer = jt.optim.AdamW(all_parameters, weight_decay=1e-2, betas=(0.9, 0.999), lr=2e-4)
    #scheduler = jt.lr_scheduler.CosineAnnealingLR(optimizer, total_epoch, eta_min=1e-6)

    best_acc_val, best_acc_test = 0., 0.
    best_epoch_val = 0


    epoch = 0


    while epoch < total_epoch:
        
        clip_model.train()
        acc_train = 0
        tot_samples = 0
        loss_epoch = 0.
        
        for (images, target) in tqdm(train_loader):
            templates_dict = load_class_names_random('text_template')
            all_templates = []
            class_indices = []
            for category_id, templates in templates_dict.items():
                all_templates.extend(templates)
                class_indices.extend([category_id] * len(templates))

            if len(all_templates) == 0:
                raise ValueError("No templates available for processing. Check templates_dict.")


            batch_size = 32 
            all_embeddings = encode_text_in_batches(clip_model, all_templates, batch_size)

            clip_weights = []
            for category_id in templates_dict.keys():
                indices = [i for i, idx in enumerate(class_indices) if idx == category_id]
                if len(indices) == 0:
                    continue
                class_embeddings = all_embeddings[indices]
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm(dim=-1, keepdim=True)

                clip_weights.append(class_embedding)

            textual_features = jt.stack(clip_weights, dim=1)

            image_features = clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            cosine_similarity = logit_scale * image_features @ textual_features

            loss_sim = nn.cross_entropy_loss(cosine_similarity, target)
            loss = loss_sim
            acc_train += cls_acc(cosine_similarity, target) * target.shape[0]
            loss_epoch += loss.item() * target.shape[0]
            tot_samples += target.shape[0]
            optimizer.step(loss)
            #scheduler.step()
        
        if epoch < total_epoch:
            acc_train /= tot_samples
            loss_epoch /= tot_samples
            # current_lr = scheduler.get_last_lr()[0]
            print('Acc: {:.4f} Loss: {:.4f}'.format(acc_train, loss_epoch))

        # Eval
        best_acc = 0
        if epoch >= 20:
            clip_model.eval()
            acc_val,acc_val1,acc_val2 = evaluate_lora(args, clip_model, val_loader)

            print("**** Val accuracy: {:.2f} {:.2f} {:.2f} ****\n".format(acc_val,acc_val1,acc_val2))
            #acc_val = evaluate_lora(args, clip_model, val_loadercar)
            #print("**** Val accuracy: {:.2f} ****\n".format(acc_val))
            if acc_val > best_acc:
                best_acc = acc_val
                save_lora(args, epoch, list_lora_layers)
        epoch += 1
    return


class Datum:
    def __init__(self, img_path, label, classname, domain):
        self.impath = img_path
        self.label = label
        self.classname = classname
        self.domain = domain


class JtDataset(Dataset):
    def __init__(self, root, split, num_shots, transform=None, transform_s=None,mode='train'):
        super(JtDataset, self).__init__()
        if mode == 'test':
            self.dataset_dir = root
        else:
            #self.dataset_dir = os.path.join(root, 'TrainSet')
            self.dataset_dir = root
        self.image_dir = self.dataset_dir
        self.split_path = os.path.join(self.dataset_dir, f'{split}.txt')
        self.classes_path = os.path.join(self.dataset_dir, 'classes.txt')
        self.mode = mode
        self.classname_to_label = self.read_classnames(self.classes_path)

        if split == 'test':
            self.data = self.read_test_split('Dataset/TestSetA')
        elif split == 'test_out':
            print(self.split_path)
            self.data = self.read_split1(self.split_path, self.image_dir)
        elif split == 'valid':
            data = self.read_split(self.split_path, self.image_dir)
            self.data = self.generate_fewshot_dataset(data, num_shots=num_shots, mode=self.mode)
        else:
            #data = self.read_split(self.split_path, 'Dataset/TrainSet')
            data = self.read_split(self.split_path, '')
            self.data = self.generate_fewshot_dataset(data, num_shots=num_shots, mode=self.mode)            
        self.transform = transform
        self.transform_s = transform_s
        self.set_attrs(total_len=len(self.data))

    def __getitem__(self, index):
        if self.mode == 'train':
            datum = self.data[index]
            img = self.read_image(datum.impath)
            img = self.transform(img)
            return img, datum.label
        else:
            datum = self.data[index]
            img = self.read_image(datum.impath)
            transformed_img = [self.transform(img)]
            transformed_imgs = [self.transform_s(img) for _ in range(127)]
            #transformed_imgs = jt.concat((transformed_img,jt.array(transformed_imgs)))
            return transformed_img, transformed_imgs, datum.label, datum.impath


    def read_image(self, path):
        if not os.path.exists(path):
            raise IOError(f'No file exists at {path}')
        img = Image.open(path).convert('RGB')
        return img

    def read_classnames(self, classes_path):
        classname_to_label = {}
        with open(classes_path, 'r') as f:
            for line in f:
                classname, label = line.strip().split()
                classname_to_label[classname] = int(label)
        return classname_to_label

    def read_split(self, split_path, image_dir):
        data = defaultdict(list)
        with open(split_path, 'r') as f:
            for line in f:
                path, label = line.strip().split()
                full_path = os.path.join(image_dir, path)
                classname = self.get_classname(int(label))
                domain = os.path.basename(os.path.dirname(full_path))
                data[int(label)].append(Datum(full_path, int(label), classname, domain))

        return [datum for label, datums in data.items() for datum in datums]

    def read_split1(self, split_path, image_dir):
        data = []
        with open(split_path, 'r') as f:
            for line in f:
                path, label = line.strip().split()
                full_path = os.path.join(image_dir, path)
                classname = self.get_classname(int(label))
                domain = os.path.basename(os.path.dirname(full_path))
                data.append(Datum(full_path, int(label), classname, domain))
        return data

    def get_classname(self, label):
        for classname, lbl in self.classname_to_label.items():
            if lbl == label:
                return classname
        return "Unknown"

    def read_test_split(self, test_dir):
        test = []
        for root, _, files in os.walk(test_dir):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    full_path = os.path.join(root, file)
                    if '__MACOSX' in full_path:
                        continue
                    domain = os.path.basename(os.path.dirname(full_path))
                    test.append(Datum(full_path, -1, "Unknown", domain))
        return test

    def generate_fewshot_dataset(self, dataset, num_shots, mode='test'):
        fewshot_dataset = []
        class_to_images = defaultdict(list)
        for datum in dataset:
            class_to_images[datum.label].append(datum)
        if self.mode == 'train':
            for label, datums in class_to_images.items():
                #random.shuffle(datums)
                selected_datums = datums
                fewshot_dataset.extend(selected_datums)
            with open('Dataset/train1.txt', 'w') as f:
                for datum in fewshot_dataset:
                    f.write(f"{datum.impath} {datum.label}\n")
        else:
            for label, datums in class_to_images.items():
                selected_datums = datums[:1]
                fewshot_dataset.extend(selected_datums)

        return fewshot_dataset


def main():
    # Load config file
    args = get_arguments()
    set_random_seed(args.seed)

    # CLIP
    # clip_model, preprocess = clip.load(args.backbone)
    clip_model, _, preprocess, _, train_tranform = clip.load("ViT-B-32.pkl")
    clip_model.eval()
    logit_scale = 100

    # Prepare dataset
    print("Preparing dataset.")

    # 初始化数据集
    root_path = 'Dataset'

    num_shots = 4

    def _convert_image_to_rgb(image):
        return image.convert("RGB")

    def _convert_image_to_tensor(image):
        """Convert a PIL image to a Jittor Var"""
        image = np.array(image, dtype=np.float32) / 255.0  # 将图像转换为 numpy 数组并归一化
        image = np.transpose(image, (2, 0, 1))  # 转换为 CHW 格式
        return jt.Var(image)

    def to_tensor(data):
        return jt.Var(data)

    class ImageToTensor(object):

        def __call__(self, input):
            input = np.asarray(input)
            if len(input.shape) < 3:
                input = np.expand_dims(input, -1)
            return to_tensor(input)


    train_tranform1 = T.Compose([
        T.RandomResizedCrop(size=224, scale=(0.05, 1)),
        T.RandomHorizontalFlip(p=0.5),
        T.ImageNormalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
        T.ToTensor()
    ])

    train_tranform2 = T.Compose([
        T.RandomResizedCrop(size=224, scale=(0.5, 1)),
        T.RandomHorizontalFlip(p=0.5),
        T.ImageNormalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
        T.ToTensor()
    ])    

    from jittor.transform import CenterCrop, ImageNormalize, Compose, _setup_size, to_pil_image, Resize

    trainset = JtDataset(root_path, 'train', num_shots=num_shots, transform=train_tranform1, mode='train')
    valset = JtDataset(root_path, 'valid1', num_shots=num_shots,  transform=preprocess, transform_s=train_tranform2, mode='test')
    testset = JtDataset(root_path, 'test', num_shots=num_shots,  transform=preprocess, transform_s=train_tranform2, mode='test')

    val_loader = DataLoader(valset, batch_size=1, num_workers=8, shuffle=False)
    test_loader = DataLoader(testset, batch_size=1, num_workers=8, shuffle=False)
    train_loader = DataLoader(trainset, batch_size=256, num_workers=8, shuffle=True)

    valsetcar = JtDataset(root_path, 'test_out', num_shots=num_shots, transform=preprocess, mode='test')
    val_loadercar = DataLoader(valsetcar, batch_size=256, num_workers=8, shuffle=False)

    run_lora(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader, val_loadercar)


if __name__ == '__main__':
    main()
