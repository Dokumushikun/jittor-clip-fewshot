import os
import random
import numpy as np
import argparse
from tqdm import tqdm
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

from jclip import clip1,clip
from jittor import attention, misc, dataset
from jittor.dataset import Dataset, DataLoader
from jittor.models.resnet import *
jt.flags.use_cuda = 1
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

IMAGENET_TEMPLATES = [
    "a photo of a {}.",
    "a photo of the {}.",
    "a sketch of a {}.",
    "a sketch of the {}.",
    "an image of a {}.",
    "an image of the {}.",
]

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



class VLPromptLearner(nn.Module):
    def __init__(self, classnames, clip_model, clip_zs):
        super(VLPromptLearner, self).__init__()
        n_cls = len(classnames)
        # Make sure Language depth >= 1

        n_ctx = 4
        ctx_init = "a photo of a"
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and n_ctx <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            prompt = clip.tokenize(ctx_init)
            with jt.no_grad():
                embedding = clip_model.token_embedding(prompt)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = jt.array(jt.randn(n_ctx, ctx_dim).numpy())
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        
        print(f"Independent V-L design")
        print(f'Initial text context: "{prompt_prefix}"')
        print(f"Number of context words (tokens) for Language prompting: {4}")
        print(f"Number of context words (tokens) for Vision prompting: {4}")
        
        self.ctx = ctx_vectors

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(clip.tokenize(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = jt.concat([clip.tokenize(p) for p in prompts], dim=0)  # (n_cls, n_tkn)
        # Also create frozen CLIP
        clip_model_temp = clip_zs
        clip_model_temp_image = clip_zs.cuda()
        with jt.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts)
            '''
            self.ZS_image_encoder = clip_zs.visual
            # Now pre-compute the frozen VL embeddings
            all_teacher_features = []
            # Using multiple text templates to ensure textual diversity during training
            for single_template in IMAGENET_TEMPLATES:
                x = [single_template.replace("{}", name) for name in classnames]
                x_tokenized = jt.concat([clip.tokenize(p) for p in x], dim=0)
                text_features = clip_model_temp.encode_text(x_tokenized)
                all_teacher_features.append(text_features.unsqueeze(1))
            '''
        #self.fixed_embeddings = jt.concat(all_teacher_features, dim=1).mean(dim=1)
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.fixed_embeddings = clip_classifier(template, clip_zs).squeeze(0)
        
        self.token_prefix = jt.array(embedding[:, :1, :])
        self.token_suffix = jt.array(embedding[:, 1 + n_ctx:, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # jt.Var
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = jt.concat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def execute(self):
        ctx = self.ctx
        if ctx.ndim == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        return prompts





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

    save_path = f'lora_weights/{epoch}_{args.filename}.pkl'
    jt.save(save_data, save_path)
    print(f'LoRA weights saved to {save_path}')


import os

def recursive_zeros_like(param_value):
    if isinstance(param_value, dict):
        return {k: recursive_zeros_like(v) for k, v in param_value.items()}
    else:
        return jt.zeros_like(param_value)
        
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
def load_lora_swa(args, list_lora_layers, swa_weights_folder):
    # 检查文件夹是否存在
    if not os.path.exists(swa_weights_folder):
        raise FileNotFoundError(f'文件夹 {swa_weights_folder} 不存在.')

    # 初始化累加权重的结构
    accumulated_weights = None
    file_count = 0

    # 遍历文件夹中的所有文件
    for filename in os.listdir(swa_weights_folder):
        load_path = os.path.join(swa_weights_folder, filename)
        print(load_path)

        if os.path.isdir(load_path):
            continue

        # 加载权重文件
        loaded_data = jt.load(load_path)
        metadata = loaded_data['metadata']

        if metadata['r'] != args.r:
            raise ValueError(f"r 不匹配:  {args.r},  {metadata['r']}")
        if metadata['alpha'] != args.alpha_lora:
            raise ValueError(f"alpha 不匹配:  {args.alpha_lora}, {metadata['alpha']}")
        if metadata['encoder'] != args.encoder:
            raise ValueError(f"编码器不匹配:  {args.encoder},  {metadata['encoder']}")
        if metadata['params'] != args.params:
            raise ValueError(f"参数不匹配: {args.params},  {metadata['params']}")
        if metadata['position'] != args.position:
            raise ValueError(f"位置不匹配:  {args.position},  {metadata['position']}")

        weights = loaded_data['weights']
        
        # 初始化累加权重
        if accumulated_weights == None:
            accumulated_weights = {
                layer_name: recursive_zeros_like(layer_weights)
                for layer_name, layer_weights in weights.items()
            }

        # 累加权重
        def recursive_add(accumulated, current):
            """递归累加嵌套的字典结构的权重."""
            for key, value in current.items():
                if isinstance(value, dict):
                    recursive_add(accumulated[key], value)
                else:
                    accumulated[key] += jt.array(value)
        
        recursive_add(accumulated_weights, weights)
        file_count += 1

    # 计算平均权重
    def recursive_average(accumulated, file_count):
        """递归计算权重的平均值."""
        for key, value in accumulated.items():
            if isinstance(value, dict):
                recursive_average(value, file_count)
            else:
                accumulated[key] /= file_count

    recursive_average(accumulated_weights, file_count)

    # 应用平均后的权重
    for i, layer in enumerate(list_lora_layers):
        layer_weights = accumulated_weights[f'layer_{i}']
        if 'q' in args.params and 'q_proj' in layer_weights:
            layer.q_proj.w_lora_A.data = layer_weights['q_proj']['w_lora_A']
            layer.q_proj.w_lora_B.data = layer_weights['q_proj']['w_lora_B']
        if 'k' in args.params and 'k_proj' in layer_weights:
            layer.k_proj.w_lora_A.data = layer_weights['k_proj']['w_lora_A']
            layer.k_proj.w_lora_B.data = layer_weights['k_proj']['w_lora_B']
        if 'v' in args.params and 'v_proj' in layer_weights:
            layer.v_proj.w_lora_A.data = layer_weights['v_proj']['w_lora_A']
            layer.v_proj.w_lora_B.data = layer_weights['v_proj']['w_lora_B']
        if 'o' in args.params and 'proj' in layer_weights:
            layer.proj.w_lora_A.data = layer_weights['proj']['w_lora_A']
            layer.proj.w_lora_B.data = layer_weights['proj']['w_lora_B']

    print(f'SWA 权重已从文件夹 {swa_weights_folder} 中加载并应用')




def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.equal(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).numpy())
    acc = 100 * acc / target.shape[0]
    return acc

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def execute(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.cast(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).cast(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[jt.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)[0]] @ self.text_projection

        return x




def set_random_seed(seed):
    # random.seed(seed)
    # np.random.seed(seed)
    misc.set_global_seed(seed)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=346373, type=int)
    # Dataset arguments
    parser.add_argument('--root_path', type=str, default='datasets')
    parser.add_argument('--dataset', type=str, default='jt')
    parser.add_argument('--shots', default=4, type=int)
    # Model arguments
    parser.add_argument('--backbone', default='ViT-B/32', type=str)
    # Training arguments
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--n_iters', default=100, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
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
                        help='file name to save the lora weights.')

    parser.add_argument('--eval_only', default=False, action='store_true',
                        help='only evaluate the LoRA modules (save_path should not be None)')

    parser.add_argument('--alpha_lora', default=1, type=int, help='scaling')

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

def clip_classifier(templates_dict, clip_model):
    with jt.no_grad():
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
    

template = load_class_names('text_template')


def evaluate_lora1(args,  moco_adapter, moco_model, channel_lp, prompt_learner, Text_encoder, clip_model, val_loader, test_loader, clip_model_zs):

    with jt.no_grad():
        text_features = clip_classifier(template, clip_model).squeeze(0)
        text_features_zs = clip_classifier(template, clip_model_zs).squeeze(0)

        tokenized_prompts = prompt_learner.tokenized_prompts
        prompts = prompt_learner()
        class_embeddings1 = Text_encoder(prompts, tokenized_prompts)
        text_features1 = class_embeddings1 / class_embeddings1.norm(dim=-1, keepdim=True)
        text_features1 =  (text_features +text_features1)/2
        text_features1 = text_features1 / text_features1.norm(dim=-1, keepdim=True)

    acc  = 0
    acc1 = 0
    acc2 = 0
    acc3 = 0
    acc4 = 0
    acc5 = 0
    acc6 = 0
    acc7 = 0
    tot_samples = 0
    

    results4 =[]
    results5 =[]
    results6 =[]
    results7 =[]
    
    with jt.no_grad():
        
        for i, (image, images, target, impath, index) in enumerate(tqdm(val_loader)):
            image = jt.array(image)
            images = jt.array(images)
            image = image.squeeze(1)
            images = images.squeeze(1)
            #print(index)
            #index = index.squeeze(0)
            #transformed_imgs = jt.stack(transformed_imgs)
            transformed_imgs = jt.concat((image, images))
            #print(transformed_imgs.shape)
            #transformed_imgs = jt.stack(transformed_imgs)
            #print(images.shape)

            image_features = clip_model.encode_image(tfm_clip(transformed_imgs))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            #cosine_similarity = image_features @ textual_features
            image_features_pt = solve_mta(image_features, text_features1.t())
            image_features_hand = solve_mta(image_features, text_features.t())
            combine_feature = (image_features_pt + image_features_hand)/2
            image_features_zs = clip_model_zs.encode_image(tfm_clip(transformed_imgs))
            image_features_zs = image_features_zs / image_features_zs.norm(dim=-1, keepdim=True)
            image_feature_zs = solve_mta(image_features_zs, text_features_zs.t())            
            #image_features_zs = all_image_features_zs[index.item()]
            logits1 = channel_lp(combine_feature)
            logits2 = channel_lp(image_feature_zs)
            logits1 = logit_normalize(logits1)
            logits2 = logit_normalize(logits2)
            #logits1 = nn.softmax(logits1)
            #logits2 = nn.softmax(logits2)
            logits = (logits1 + logits2)/2
            logits = logit_normalize(logits)   
            
            aux_logits = moco_model(tfm_moco(image))
            out_moco = moco_adapter(aux_logits)
            out_moco = logit_normalize(out_moco)
            
            #out_moco = nn.softmax(out_moco_)
            cosine_similarity =  100.*image_features_hand @ text_features.t()
            cosine_similarity1 = 100.* image_features_pt @ text_features1.t()
            cosine_similarity3 =  100.*image_feature_zs @ text_features_zs.t()

            cosine_similarity2 = (cosine_similarity + cosine_similarity1)/2
            cosine_similarity4 = (cosine_similarity2 + cosine_similarity3)/2
            #cosine_similarity5 = (cosine_similarity2 + cosine_similarity3)/2 + 0.5 * logits
            cosine_similarity5 = (cosine_similarity2 + cosine_similarity3)/2 + 0.5 *logits
            cosine_similarity6 = (cosine_similarity2 + cosine_similarity3)/2 + 0.5 *out_moco
            cosine_similarity7 = (cosine_similarity2 + cosine_similarity3)/2 + 0.5 *(logits + out_moco) / 2

            # 获取 top-5 预测结果的索引

            top5_prob, top5_pred4 = cosine_similarity4.topk(5, dim=-1)
            top5_prob, top5_pred5 = cosine_similarity5.topk(5, dim=-1)
            top5_prob, top5_pred6 = cosine_similarity6.topk(5, dim=-1)
            top5_prob, top5_pred7 = cosine_similarity7.topk(5, dim=-1)
            for idx in range(1):
                top5_labels4 = top5_pred4[idx].tolist()  
                top5_str4 = ' '.join(map(str, top5_labels4)) 
                results4.append(f"{impath} {top5_str4}")
                
                top5_labels5 = top5_pred5[idx].tolist()  
                top5_str5 = ' '.join(map(str, top5_labels5)) 
                results5.append(f"{impath} {top5_str5}")

                top5_labels6 = top5_pred6[idx].tolist()  
                top5_str6 = ' '.join(map(str, top5_labels6)) 
                results6.append(f"{impath} {top5_str6}")
                
                top5_labels7 = top5_pred7[idx].tolist()  
                top5_str7 = ' '.join(map(str, top5_labels7)) 
                results7.append(f"{impath} {top5_str7}")


        with open(f'final_results/top5_results4.txt', 'w') as f:
            for result in results4:
                f.write(result + '\n')
        with open(f'final_results/top5_results5.txt', 'w') as f:
            for result in results5:
                f.write(result + '\n')
        with open(f'final_results/top5_results6.txt', 'w') as f:
            for result in results6:
                f.write(result + '\n')   
        with open(f'final_results/top5_results7.txt', 'w') as f:
            for result in results7:
                f.write(result + '\n')  
        

        for i, (image, images, target, impath, index) in enumerate(tqdm(val_loader)):
            image = jt.array(image)
            images = jt.array(images)
            image = image.squeeze(1)
            images = images.squeeze(1)
            #print(index)
            #index = index.squeeze(0)
            #transformed_imgs = jt.stack(transformed_imgs)
            transformed_imgs = jt.concat((image, images))
            #print(transformed_imgs.shape)
            #transformed_imgs = jt.stack(transformed_imgs)
            #print(images.shape)
            image_features = clip_model.encode_image(tfm_clip(transformed_imgs))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            #cosine_similarity = image_features @ textual_features
            image_features_pt = solve_mta(image_features, text_features1.t())
            image_features_hand = solve_mta(image_features, text_features.t())
            combine_feature = (image_features_pt + image_features_hand)/2

            image_features_zs = clip_model_zs.encode_image(tfm_clip(transformed_imgs))
            image_features_zs = image_features_zs / image_features_zs.norm(dim=-1, keepdim=True)
            image_feature_zs = solve_mta(image_features_zs, text_features_zs.t())            
            #image_features_zs = all_image_features_zs[index.item()]
            logits = channel_lp(image_features)
            #logits2 = channel_lp(image_feature_zs)
            #logits1 = logit_normalize(logits1)
            #logits2 = logit_normalize(logits2)
            #logits1 = nn.softmax(logits1)
            #logits2 = nn.softmax(logits2)
            #logits = (logits1 + logits2)/2
            logits = logit_normalize(logits).mean(dim=0)   

            aux_logits = moco_model(tfm_moco(transformed_imgs))
            out_moco = moco_adapter(aux_logits)
            out_moco = logit_normalize(out_moco).mean(dim=0)
            
            #out_moco = nn.softmax(out_moco_)
            cosine_similarity =  100.* image_features_hand @ text_features.t()
            cosine_similarity1 =  100.* image_features_pt @ text_features1.t()
            cosine_similarity3 =  100.* image_feature_zs @ text_features_zs.t()

            cosine_similarity2 = (cosine_similarity + cosine_similarity1)/2
            cosine_similarity4 = (cosine_similarity2 + cosine_similarity3)/2
            #cosine_similarity5 = (cosine_similarity2 + cosine_similarity3)/2 + 0.5 * logits
            cosine_similarity5 = (cosine_similarity2 + cosine_similarity3)/2 + 0.5 *logits
            cosine_similarity6 = (cosine_similarity2 + cosine_similarity3)/2 + 0.5 *out_moco
            cosine_similarity7 = (cosine_similarity2 + cosine_similarity3)/2 + 0.5 *(logits + out_moco) / 2

            acc += cls_acc(cosine_similarity, target) * len(cosine_similarity)
            acc1 += cls_acc(cosine_similarity1, target) * len(cosine_similarity)
            acc2 += cls_acc(cosine_similarity2, target) * len(cosine_similarity)            
            acc3 += cls_acc(cosine_similarity3, target) * len(cosine_similarity)
            
            acc4 += cls_acc(cosine_similarity4, target) * len(cosine_similarity)#combine + zslora
            acc5 += cls_acc(cosine_similarity5, target) * len(cosine_similarity)#+lp
            acc6 += cls_acc(cosine_similarity6, target) * len(cosine_similarity)#+moco
            acc7 += cls_acc(cosine_similarity7, target) * len(cosine_similarity)#+lp moco combine
            tot_samples += len(cosine_similarity)
            #print(acc5,acc6,acc7)

        acc /= tot_samples
        acc1 /= tot_samples 
        acc2 /= tot_samples
        acc3 /= tot_samples 
        acc4 /= tot_samples 
        acc5 /= tot_samples
        acc6 /= tot_samples 
        acc7 /= tot_samples 

    return acc,acc1,acc2,acc3,acc4,acc5,acc6,acc7

def evaluate_lora2(args, moco_adapter, moco_model, channel_lp, prompt_learner, Text_encoder, clip_model, val_loader, test_loader, clip_model_zs):

    with jt.no_grad():
        text_features_zs = clip_classifier(template, clip_model_zs).squeeze(0)
    acc  = 0

    tot_samples = 0
    results =[]
    with jt.no_grad():
        for i, (image, images, target, impath, index) in enumerate(tqdm(val_loader)):
            image = jt.array(image)
            images = jt.array(images)
            image = image.squeeze(1)
            images = images.squeeze(1)
            transformed_imgs = jt.concat((image, images))
            image_features_zs = clip_model_zs.encode_image(tfm_clip(transformed_imgs))
            image_features_zs = image_features_zs / image_features_zs.norm(dim=-1, keepdim=True)
            image_feature_zs = solve_mta(image_features_zs, text_features_zs.t())            

            cosine_similarity =  100.* image_feature_zs @ text_features_zs.t()
            acc += cls_acc(cosine_similarity, target) * len(cosine_similarity)
            tot_samples += len(cosine_similarity)
            top5_prob, top5_pred = cosine_similarity.topk(5, dim=-1)
            for idx in range(1):
                top5_labels = top5_pred[idx].tolist()  
                top5_str = ' '.join(map(str, top5_labels)) 
                results.append(f"{impath} {top5_str}")
        acc /= tot_samples
        with open(f'final_results/top5_results_ood.txt', 'w') as f:
            for result in results:
                f.write(result + '\n')  
    return acc

def pre_load_features(clip_model, loader):
    features, labels = [], []
    with jt.no_grad():
        for i, (images, target) in enumerate(tqdm(loader)):
            image_features = clip_model.encode_image(tfm_clip(images))
            image_features /= image_features.norm(dim=-1, keepdim=True)
            features.append(image_features)
            labels.append(target)
        features, labels = jt.concat(features), jt.concat(labels)

    return features, labels

def pre_load_features_moco(moco_model, loader):
    moco_features = []
    moco_labels = []
    with jt.no_grad():
        for augment_idx in range(1):
            moco_features_current = []
            for i, (images, target, index) in enumerate(tqdm(loader)):
                image_features = moco_model(tfm_moco(images))
                moco_features_current.append(image_features)
                moco_labels.append(target)
            moco_features.append(jt.concat(moco_features_current, dim=0).unsqueeze(0))

    moco_features = jt.concat(moco_features, dim=0).mean(dim=0)
    moco_features /= moco_features.norm(dim=-1, keepdim=True)

    moco_labels = jt.concat(moco_labels)

    return moco_features, moco_labels
    

def kl_div(log_probs, target_log_probs, reduction='sum'):
    kl_div = jt.exp(target_log_probs) * (target_log_probs - log_probs)
    if reduction == 'sum':
        return kl_div.sum()
    elif reduction == 'mean':
        return kl_div.mean()
    else:
        return kl_div
        
def get_vlt_parameters(model):
    params = []
    for name, param in model.named_parameters():
        if 'VPT' in name:
            params.append(param)
    return params
    
def get_prompt_parameters(model):
    params = []
    for name, param in model.named_parameters():
        if "ZS_image_encoder" not in name:
            params.append(param)
    return params


class Channel_LP(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale1 = jt.ones(512)
        self.bias1 = jt.zeros(512)
        self.fc = nn.Linear(512, 403)
    def execute(self, features):
        scale1 = self.scale1.unsqueeze(0)
        bias1 = self.bias1.unsqueeze(0)
        features = scale1 * features + bias1
        out = self.fc(features)
        return out

class Moco_Adapter(nn.Module):
    def __init__(self):
        super().__init__()
        #self.scale1 = jt.ones(2048)
        #self.bias1 = jt.zeros(2048)
        self.fc = nn.Linear(2048, 403)
    def execute(self, features):
        #scale1 = self.scale1.unsqueeze(0)
        #bias1 = self.bias1.unsqueeze(0)
        #features = scale1 * features + bias1
        out = self.fc(features)
        return out

def load_class_names_random(filepath,idx):
    filename = 'text_template' + str(idx) + '.txt'
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
    
def load_moco(pretrain_path):
    print("=> creating model")
    model = resnet50(pretrained=False)  # 不加载预训练权重
    linear_keyword = 'fc'
    if os.path.isfile(pretrain_path):
        print("=> loading checkpoint '{}'".format(pretrain_path))
        checkpoint = jt.load(pretrain_path)
        state_dict = checkpoint["state_dict"]
        new_state_dict = {}
        for k, v in state_dict.items():
            '''
            if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % linear_keyword):
                # remove prefix
                new_key = k[len("module.base_encoder."):]
                new_state_dict[new_key] = v
            else:
                new_state_dict[k] = v
            '''
            if k.startswith('base_encoder') and not k.startswith('base_encoder.%s' % linear_keyword):
                # remove prefix
                new_key = k[len("base_encoder."):]
                new_state_dict[new_key] = v
            else:
                new_state_dict[k] = v


        model.load_parameters(new_state_dict)
        print("=> loaded pre-trained model '{}'".format(pretrain_path))
    else:
        print("=> no checkpoint found at '{}'".format(pretrain_path))
        raise FileNotFoundError

    # 将全连接层设为Identity
    model.fc = jt.nn.Identity()
    return model, 2048
    
tfm_clip = T.Compose([T.ImageNormalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
tfm_moco = T.Compose([T.ImageNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def logit_normalize(logit):
    logits_std = jt.std(logit)
    logits_mean = jt.mean(logit, dim=1, keepdim=True)
    logit = (logit - logits_mean) / logits_std
    return logit

def gaussian_kernel(mu, bandwidth, datapoints):
    dist = jt.norm(datapoints - mu, dim=-1, p=2)
    density = jt.exp(-dist**2/(2*bandwidth**2))
    return density
def cdist(x1, x2):
    x1_square = jt.sum(x1 ** 2, dim=1, keepdims=True)
    x2_square = jt.sum(x2 ** 2, dim=1, keepdims=True)
    dist = jt.sqrt(x1_square - 2 * jt.matmul(x1, x2.transpose()) + x2_square.transpose())
    return dist

def solve_mta1(image_features,text_features):

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

    #output = mode.unsqueeze(0) @ text_features * 100

    return mode.unsqueeze(0)

def pre_load_zs(clip_model_zs, trainset_feature_loader):
    features, labels = [], []
    text_features_zs = clip_classifier(template, clip_model_zs).squeeze(0)
    with jt.no_grad():
        for i, (image, images, target, _, _) in enumerate(tqdm(trainset_feature_loader)):
            image = jt.array(image)
            images = jt.array(images)
            image = image.squeeze(1)
            images = images.squeeze(1)
            transformed_imgs = jt.concat((image, images))
            image_features = clip_model_zs.encode_image(tfm_clip(transformed_imgs))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            #print(image_features.shape)
            image_feature = solve_mta(image_features, text_features_zs.t())
            #image_feature = image_features.mean(0)
            features.append(image_feature)
            labels.append(target)
    jt.save(features, "features_zs1.pkl")
    print(f"Features and labels have been saved.")
    return features, labels

def pre_load_zs1(clip_model_zs, trainset_feature_loader):
    features, labels = [], []
    text_features_zs = clip_classifier(template, clip_model_zs).squeeze(0)
    with jt.no_grad():
        for i, (image, images, target, _, _) in enumerate(tqdm(trainset_feature_loader)):
            image = jt.array(image)
            images = jt.array(images)
            image = image.squeeze(1)
            images = images.squeeze(1)
            transformed_imgs = jt.concat((image, images))
            image_features = clip_model_zs.encode_image(tfm_clip(transformed_imgs))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            #print(image_features.shape)
            image_feature = solve_mta(image_features, text_features_zs.t())
            #image_feature = image_features.mean(0)
            features.append(image_feature)
            labels.append(target)
    
    jt.save(features, "features_zs2.pkl")
    jt.save(jt.array(labels), "label_zs2.pkl")
    print(f"Features and labels have been saved.")
    return features, labels
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

        if split == 'TestSetB_1':
            data = self.read_split_test(self.split_path, '')
            self.data = self.generate_fewshot_dataset(data, num_shots=num_shots, mode=self.mode)
        elif split == 'TestSetB_2':
            data = self.read_split_test(self.split_path, '')
            self.data = self.generate_fewshot_dataset(data, num_shots=num_shots, mode=self.mode)
        elif split == 'test_out':
            data = self.read_split(self.split_path, 'Dataset')
            self.data = self.generate_fewshot_dataset(data, num_shots=num_shots, mode=self.mode)
        elif split == 'valid':
            data = self.read_split(self.split_path, 'Dataset')
            self.data = self.generate_fewshot_dataset(data, num_shots=num_shots, mode=self.mode)
        else:
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
            return img, datum.label, index
        else:
            datum = self.data[index]
            img = self.read_image(datum.impath)
            transformed_img = [self.transform(img)]

            transformed_imgs = [self.transform_s(img) for _ in range(512)]
            #transformed_imgs = jt.concat((transformed_img,jt.array(transformed_imgs)))
            return transformed_img, transformed_imgs, datum.label, datum.impath, index


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

    def read_split_test(self, split_path, image_dir):
        data = []
        with open(split_path, 'r') as f:
            for line in f:
                path = line.strip()
                full_path = os.path.join(image_dir, path)
                domain = os.path.basename(os.path.dirname(full_path))
                data.append(Datum(full_path, -1, "Unknown", domain))
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
        for label, datums in class_to_images.items():

            if self.mode == 'train':
                #random.shuffle(datums)
                selected_datums = datums
                fewshot_dataset.extend(selected_datums)
            else:
                selected_datums = datums
                fewshot_dataset.extend(selected_datums)
        return fewshot_dataset

def classnames_(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    classnames = []
    for line in lines:
        # 分割行，提取类别部分
        parts = line.split()
        classname_with_prefix = parts[0]
        
        # 去掉前缀和下划线，保留类别名
        classname = '_'.join(classname_with_prefix.split('_')[1:])
        classnames.append(classname)

    return classnames

def load_txt_to_dict(file_path):
    data_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            key = parts[0]  
            values = parts[1:] 
            data_dict[key] = values
    return data_dict

def save_dict_to_txt(data_dict, file_path):
    with open(file_path, 'w') as file:
        for key, values in data_dict.items():
            line = f"{key} {' '.join(values)}\n"
            file.write(line)

def update_txt_file(base_txt, update_txt):

    base_dict = load_txt_to_dict(base_txt)
    update_dict = load_txt_to_dict(update_txt)


    base_dict.update(update_dict)

    save_dict_to_txt(base_dict, base_txt)
def evaluate_base(args,  moco_adapter, moco_model, channel_lp, prompt_learner, Text_encoder, clip_model, val_loader, test_loader, clip_model_zs):

    with jt.no_grad():
        text_features = clip_classifier(template, clip_model).squeeze(0)
        text_features_zs = clip_classifier(template, clip_model_zs).squeeze(0)

        tokenized_prompts = prompt_learner.tokenized_prompts
        prompts = prompt_learner()
        class_embeddings1 = Text_encoder(prompts, tokenized_prompts)
        text_features1 = class_embeddings1 / class_embeddings1.norm(dim=-1, keepdim=True)
        text_features1 =  (text_features +text_features1)/2
        text_features1 = text_features1 / text_features1.norm(dim=-1, keepdim=True)
    
    results6 =[]
    
    with jt.no_grad():
        
        for i, (image, images, target, impath, index) in enumerate(tqdm(test_loader)):
            image = jt.array(image)
            images = jt.array(images)
            image = image.squeeze(1)
            images = images.squeeze(1)
            #print(index)
            #index = index.squeeze(0)
            #transformed_imgs = jt.stack(transformed_imgs)
            transformed_imgs = jt.concat((image, images))
            #print(transformed_imgs.shape)
            #transformed_imgs = jt.stack(transformed_imgs)
            #print(images.shape)

            image_features = clip_model.encode_image(tfm_clip(transformed_imgs))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            #cosine_similarity = image_features @ textual_features
            image_features_pt = solve_mta(image_features, text_features1.t())
            image_features_hand = solve_mta(image_features, text_features.t())
            combine_feature = (image_features_pt + image_features_hand)/2
            image_features_zs = clip_model_zs.encode_image(tfm_clip(transformed_imgs))
            image_features_zs = image_features_zs / image_features_zs.norm(dim=-1, keepdim=True)
            image_feature_zs = solve_mta(image_features_zs, text_features_zs.t())            
            #image_features_zs = all_image_features_zs[index.item()]
            logits1 = channel_lp(combine_feature)
            logits2 = channel_lp(image_feature_zs)
            logits1 = logit_normalize(logits1)
            logits2 = logit_normalize(logits2)
            #logits1 = nn.softmax(logits1)
            #logits2 = nn.softmax(logits2)
            logits = (logits1 + logits2)/2
            logits = logit_normalize(logits)   
            
            aux_logits = moco_model(tfm_moco(image))
            out_moco = moco_adapter(aux_logits)
            out_moco = logit_normalize(out_moco)
            
            #out_moco = nn.softmax(out_moco_)
            cosine_similarity =  100.*image_features_hand @ text_features.t()
            cosine_similarity1 = 100.* image_features_pt @ text_features1.t()
            cosine_similarity3 =  100.*image_feature_zs @ text_features_zs.t()

            cosine_similarity2 = (cosine_similarity + cosine_similarity1)/2
            cosine_similarity4 = (cosine_similarity2 + cosine_similarity3)/2
            cosine_similarity5 = (cosine_similarity2 + cosine_similarity3)/2 + 0.5 *logits
            cosine_similarity6 = (cosine_similarity2 + cosine_similarity3)/2 + 0.5 *out_moco

            top5_prob, top5_pred6 = cosine_similarity6.topk(5, dim=-1)
            for idx in range(1):
                top5_labels6 = top5_pred6[idx].tolist()  
                top5_str6 = ' '.join(map(str, top5_labels6)) 
                results6.append(f"{impath} {top5_str6}")

        with open(f'final_results/top5_results6.txt', 'w') as f:
            for result in results6:
                f.write(result + '\n')   

def evaluate_new(args, moco_adapter, moco_model, channel_lp, prompt_learner, Text_encoder, clip_model, val_loader, test_loader, clip_model_zs):

    with jt.no_grad():
        text_features = clip_classifier(template, clip_model).squeeze(0)
        text_features_zs = clip_classifier(template, clip_model_zs).squeeze(0)
    acc  = 0

    tot_samples = 0
    results =[]
    with jt.no_grad():
        for i, (image, images, target, impath, index) in enumerate(tqdm(test_loader)):
            image = jt.array(image)
            images = jt.array(images)
            image = image.squeeze(1)
            images = images.squeeze(1)

            transformed_imgs = jt.concat((image, images))
            image_features_zs = clip_model_zs.encode_image(tfm_clip(transformed_imgs))
            image_features_zs = image_features_zs / image_features_zs.norm(dim=-1, keepdim=True)
            image_feature_zs = solve_mta(image_features_zs, text_features_zs.t())            

            cosine_similarity =  100.* image_feature_zs @ text_features_zs.t()
            
            tot_samples += len(cosine_similarity)
            
            top5_prob, top5_pred = cosine_similarity.topk(5, dim=-1)

            for idx in range(1):
                top5_labels = top5_pred[idx].tolist()  
                top5_str = ' '.join(map(str, top5_labels)) 
                results.append(f"{impath} {top5_str}")
                
        
        with open(f'final_results/top5_results_ood.txt', 'w') as f:
            for result in results:
                f.write(result + '\n')  
    return acc
    
def run_test(args, clip_model_zs, clip_model, logit_scale, dataset, val_loader1, val_loader2, test_loader1, test_loader2):

    list_lora_layers1 = apply_lora(args, clip_model_zs)
    load_lora(args, list_lora_layers1, 'lora_weights1/lora_weights.pkl')
    list_lora_layers = apply_lora(args, clip_model)
    
    classnames = classnames_('classes.txt')
    prompt_learner = VLPromptLearner(classnames, clip_model, clip_model_zs)
    tokenized_prompts = prompt_learner.tokenized_prompts

    Text_encoder = TextEncoder(clip_model)
    
    moco_model, args.feat_dim = load_moco("r-50-1000ep.pkl")
    
    channel_lp = Channel_LP()


    moco_adapter = Moco_Adapter() 
    
    moco_adapter.load('test_pkl/moco_adapter.pkl')
    channel_lp.load('test_pkl/channel.pkl')
    clip_model.load('test_pkl/clip_model.pkl')
    prompt_learner.load('test_pkl/PromptLearner.pkl')
    load_lora(args, list_lora_layers, 'test_pkl/lora_weights.pkl')
    clip_model_ori, _, _ , _, _ = clip.load("ViT-B-32.pkl")
    clip_model_zs.eval()
    moco_adapter.eval()
    moco_model.eval()
    channel_lp.eval()
    clip_model.eval()
    clip_model_ori.eval()   
    acc_val = evaluate_lora2(args, moco_adapter, moco_model, channel_lp, prompt_learner, Text_encoder, clip_model, val_loader1, test_loader2, clip_model_ori)
    print(acc_val)
    acc_val,acc_val1,acc_val2,acc_val3,acc_val4,acc_val5,acc_val6 ,acc_val7 = evaluate_lora1(args, moco_adapter, moco_model, channel_lp, prompt_learner, Text_encoder, clip_model, val_loader2, test_loader1, clip_model_zs)
    print(acc_val,acc_val1,acc_val2,acc_val3,acc_val4,acc_val5,acc_val6,acc_val7)    

    


    base_txt = 'final_results/top5_results6.txt'
    update_txt = 'final_results/top5_results_ood.txt'

    update_txt_file(base_txt, update_txt)
import re
def process_line(line):
    # 使用正则表达式提取文件名
    match = re.search(r"\['(.*?)'\]", line)
    if match:
        # 提取文件名
        file_name = match.group(1).split('/')[-1]
        # 用文件名替换整行
        line = line.replace(match.group(0), file_name)
    return line
    
def run_test1(args, clip_model_zs, clip_model, logit_scale, dataset, val_loader1, val_loader2, test_loader1, test_loader2):

    list_lora_layers1 = apply_lora(args, clip_model_zs)
    load_lora(args, list_lora_layers1, 'lora_weights1/lora_weights.pkl')
    
    list_lora_layers = apply_lora(args, clip_model)
    
    classnames = classnames_('classes.txt')
    prompt_learner = VLPromptLearner(classnames, clip_model, clip_model_zs)
    tokenized_prompts = prompt_learner.tokenized_prompts


    
    moco_model, args.feat_dim = load_moco("r-50-1000ep.pkl")
    
    channel_lp = Channel_LP()


    moco_adapter = Moco_Adapter() 
    
    moco_adapter.load('test_pkl/moco_adapter.pkl')
    channel_lp.load('test_pkl/channel.pkl')
    clip_model.load('test_pkl/clip_model.pkl')
    prompt_learner.load('test_pkl/PromptLearner.pkl')
    load_lora(args, list_lora_layers, 'test_pkl/lora_weights.pkl')
    Text_encoder = TextEncoder(clip_model)
    
    clip_model_ori, _, _ , _, _ = clip.load("ViT-B-32.pkl")
    clip_model_zs.eval()
    moco_adapter.eval()
    moco_model.eval()
    channel_lp.eval()
    clip_model.eval()
    clip_model_ori.eval()   

    evaluate_base(args, moco_adapter, moco_model, channel_lp, prompt_learner, Text_encoder, clip_model, val_loader2, test_loader1, clip_model_zs)

    evaluate_new(args, moco_adapter, moco_model, channel_lp, prompt_learner, Text_encoder, clip_model, val_loader1, test_loader2, clip_model_ori)

    base_txt = 'final_results/top5_results6.txt'
    update_txt = 'final_results/top5_results_ood.txt'

    update_txt_file(base_txt, update_txt)


    input_file = 'final_results/top5_results6.txt'
    output_file = 'result.txt'  

    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            processed_line = process_line(line)
            outfile.write(processed_line)

def main():

    args = get_arguments()
    set_random_seed(args.seed)
    clip_model, preprocess, _, _, train_tranform = clip1.load_vlp("ViT-B-32.pkl")
    #clip_model, preprocess, _, _, train_tranform = clip.load("ViT-B-32.pkl")
    clip_model_zs, _, _ , _, _ = clip.load("ViT-B-32.pkl")

    logit_scale = 100
    total_epochs = 50
    
    clip_model.eval()

    print("Preparing dataset.")


    root_path = 'Dataset'
    num_shots = 4

    def _convert_image_to_rgb(image):
        return image.convert("RGB")

    def _convert_image_to_tensor(image):
        """Convert a PIL image to a Jittor Var"""
        image = np.array(image, dtype=np.float32) / 255.0  
        image = np.transpose(image, (2, 0, 1)) 
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
        #T.ImageNormalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
        T.ToTensor()
    ])

    train_tranform2 = T.Compose([
        T.RandomResizedCrop(size=224, scale=(0.2, 1)),
        T.RandomHorizontalFlip(p=0.5),
        #T.ImageNormalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
        T.ToTensor()
    ])

    from jittor.transform import CenterCrop, ImageNormalize, Compose, _setup_size, to_pil_image, Resize
    '''
    valset2 = JtDataset(root_path, 'valid', num_shots=num_shots, transform=preprocess, transform_s=train_tranform2, mode='test')
    val_loader2 = DataLoader(valset2, batch_size=1, num_workers=8, shuffle=False)

    valset1 = JtDataset(root_path, 'test_out', num_shots=num_shots, transform=preprocess, transform_s=train_tranform2, mode='test')
    val_loader1 = DataLoader(valset1, batch_size=1, num_workers=8, shuffle=False)
    '''
    testset1 = JtDataset(root_path, 'TestSetB_1', num_shots=num_shots, transform=preprocess, transform_s=train_tranform2, mode='test')
    test_loader1 = DataLoader(testset1, batch_size=1, num_workers=8, shuffle=False)
    testset2 = JtDataset(root_path, 'TestSetB_2', num_shots=num_shots, transform=preprocess, transform_s=train_tranform2, mode='test')
    test_loader2 = DataLoader(testset2, batch_size=1, num_workers=8, shuffle=False)
    #valsetcar = JtDataset(root_path, 'test_out', num_shots=num_shots, transform=preprocess, transform_s=train_tranform2, mode='test')
    #val_loadercar = DataLoader(valsetcar, batch_size=256, num_workers=8, shuffle=False)
    run_test(args, clip_model_zs, clip_model, logit_scale, dataset, val_loader1, val_loader2, test_loader1, test_loader2)
    #run_test1(args, clip_model_zs, clip_model, logit_scale, dataset, val_loader1, val_loader2, test_loader1, test_loader2)


if __name__ == '__main__':
    main()
