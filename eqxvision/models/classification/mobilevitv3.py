from typing import Optional, Dict, Tuple, Union, Any
import argparse
from eqxvision.models.classification.config_mobilevitv3 import get_configuration
import equinox as eqx
from eqxvision.layers import ConvLayer
from eqxvision.layers.transformer import TransformerEncoder ,get_normalization_layer#, LinearLayer, GlobalPool, Identity
import jax 
from jax import numpy as jnp
import math 

def make_divisible(
    v: Union[float, int],
    divisor: Optional[int] = 8,
    min_value: Optional[Union[float, int]] = None,
) -> Union[float, int]:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class BaseEncoder(eqx.Module):
    
    conv_1: eqx.Module
    layer_1: eqx.Module
    layer_2: eqx.Module
    layer_3: eqx.Module
    layer_4: eqx.Module
    layer_5: eqx.Module
    conv_1x1_exp: eqx.Module
    classifier: eqx.Module
    
    round_nearest: Any = eqx.field(static=True)
    dilation: Any = eqx.field(static=True)
    dilate_l4: Any = eqx.field(static=True)
    dilate_l5: Any = eqx.field(static=True)
    model_conf_dict: Any = eqx.field(static=True)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        
        self.conv_1 = None
        self.layer_1 = None
        self.layer_2 = None
        self.layer_3 = None
        self.layer_4 = None
        self.layer_5 = None
        self.conv_1x1_exp = None
        self.classifier = None
        
        self.round_nearest = 8
        
        self.dilation = 1
        output_stride = kwargs.get("output_stride", None)
        self.dilate_l4 = False
        self.dilate_l5 = False
        if output_stride == 8:
            self.dilate_l4 = True
            self.dilate_l5 = True
        elif output_stride == 16:
            self.dilate_l5 = True

        self.model_conf_dict = dict()
        
    # def extract_features(self, x, *args, **kwargs):
    #     x = self.conv_1(x)
    #     x = self.layer_1(x)
    #     x = self.layer_2(x)
    #     x = self.layer_3(x)

    #     x = self.layer_4(x)
    #     x = self.layer_5(x)
    #     x = self.conv_1x1_exp(x)
    #     return x
    
    # def forward(self, x, *args, **kwargs):
    #     x = self.extract_features(x)
    #     x = self.classifier(x)
    #     return x
    
class InvertedResidual(eqx.nn.StatefulLayer):
    block: eqx.Module
    use_res_connect: Any = eqx.field(static=True)
    
    def __init__(
        self,
        opts,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: Union[int, float],
        dilation: int = 1,
        skip_connection: Optional[bool] = True,
        key = None,
        *args,
        **kwargs
    ) -> None:
        assert stride in [1, 2]
        hidden_dim = make_divisible(int(round(in_channels * expand_ratio)), 8)

        super().__init__()

        block = []
        # print(expand_ratio)
        keys = jax.random.split(key, 3)
        if expand_ratio != 1:
            block.append(
                ConvLayer(
                    opts,
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=1,
                    use_act=True,
                    use_norm=True,
                    key=keys[0],
                ),
            )

            block.append(
                ConvLayer(
                opts,
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                stride=stride,
                kernel_size=3,
                groups=hidden_dim,
                use_act=True,
                use_norm=True,
                dilation=dilation,
                key=keys[1],
            ),
        )

            block.append(
                ConvLayer(
                opts,
                in_channels=hidden_dim,
                out_channels=out_channels,
                kernel_size=1,
                use_act=False,
                use_norm=True,
                key=keys[2],
            ),
        )

        self.block = eqx.nn.Sequential(block)
        # self.in_channels = in_channels
        # self.out_channels = out_channels
        # self.exp = expand_ratio
        # self.dilation = dilation
        # self.stride = stride
        self.use_res_connect = (
            stride == 1 and in_channels == out_channels and skip_connection
        )

    def __call__(self, x, state, key):
        # print(x.shape)
        if self.use_res_connect:
            x_new, state = self.block(x, state, key=key)
            return x + x_new, state
        else:
            return self.block(x, state, key=key)
        

class MobileViTv3Block(eqx.nn.StatefulLayer):
    
    local_rep: eqx.Module
    global_rep: eqx.Module
    global_rep_out: eqx.Module
    conv_proj: eqx.Module
    fusion: eqx.Module
    
    
    patch_h: Any = eqx.field(static=True)
    patch_w: Any = eqx.field(static=True)
    patch_area: Any = eqx.field(static=True)
    
    """
        MobileViTv3 block
    """
    def __init__(self, opts, in_channels: int, transformer_dim: int, ffn_dim: int,
                 n_transformer_blocks: Optional[int] = 2,
                 head_dim: Optional[int] = 32, attn_dropout: Optional[float] = 0.1,
                 dropout: Optional[int] = 0.1, ffn_dropout: Optional[int] = 0.1, patch_h: Optional[int] = 8,
                 patch_w: Optional[int] = 8, transformer_norm_layer: Optional[str] = "layer_norm",
                 conv_ksize: Optional[int] = 3,
                 dilation: Optional[int] = 1, var_ffn: Optional[bool] = False,
                 no_fusion: Optional[bool] = False,
                 key = False,
                 *args, **kwargs):
        
        keys = jax.random.split(key, 5)

        # For MobileViTv3: Normal 3x3 convolution --> Depthwise 3x3 convolution
        conv_3x3_in = ConvLayer(
            opts=opts, in_channels=in_channels, out_channels=in_channels,
            kernel_size=conv_ksize, stride=1, use_norm=True, use_act=True, dilation=dilation,
            groups=in_channels, key=keys[0]
        )
        conv_1x1_in = ConvLayer(
            opts=opts, in_channels=in_channels, out_channels=transformer_dim,
            kernel_size=1, stride=1, use_norm=False, use_act=False, key=keys[1]
        )


        conv_1x1_out = ConvLayer(
            opts=opts, in_channels=transformer_dim, out_channels=in_channels,
            kernel_size=1, stride=1, use_norm=True, use_act=True, key=keys[2]
        )
        conv_3x3_out = None

        # For MobileViTv3: input+global --> local+global
        if not no_fusion:
            #input_ch = tr_dim + in_ch
            conv_3x3_out = ConvLayer(
                opts=opts, in_channels= transformer_dim + in_channels, out_channels=in_channels,
                kernel_size=1, stride=1, use_norm=True, use_act=True, key=keys[3]
            )

        super(MobileViTv3Block, self).__init__()
        local_rep = []
        local_rep.append(conv_3x3_in)
        local_rep.append(conv_1x1_in)
        self.local_rep = eqx.nn.Sequential(local_rep)

        assert transformer_dim % head_dim == 0
        num_heads = transformer_dim // head_dim
        ffn_dims = [ffn_dim] * n_transformer_blocks

        keys = jax.random.split(keys[-1],n_transformer_blocks)
        global_rep = [
            TransformerEncoder(opts=opts, embed_dim=transformer_dim, ffn_latent_dim=ffn_dims[block_idx], num_heads=num_heads,
                               attn_dropout=attn_dropout, dropout=dropout, ffn_dropout=ffn_dropout,
                               transformer_norm_layer=transformer_norm_layer, key=keys[block_idx])
            for block_idx in range(n_transformer_blocks)
        ]
        
        global_rep_out = get_normalization_layer(opts=opts, norm_type=transformer_norm_layer, num_features=transformer_dim)
        self.global_rep_out = global_rep_out
        # global_rep.append(
            
        # )
        self.global_rep = eqx.nn.Sequential(global_rep)

        self.conv_proj = conv_1x1_out

        self.fusion = conv_3x3_out

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_area = self.patch_w * self.patch_h

        # self.cnn_in_dim = in_channels
        # self.cnn_out_dim = transformer_dim
        # self.n_heads = num_heads
        # self.ffn_dim = ffn_dim
        # self.dropout = dropout
        # self.attn_dropout = attn_dropout
        # self.ffn_dropout = ffn_dropout
        # self.dilation = dilation
        # self.ffn_max_dim = ffn_dims[0]
        # self.ffn_min_dim = ffn_dims[-1]
        # self.var_ffn = var_ffn
        # self.n_blocks = n_transformer_blocks
        # self.conv_ksize = conv_ksize


    def unfolding(self, feature_map):
        patch_w, patch_h = self.patch_w, self.patch_h
        patch_area = int(patch_w * patch_h)
        in_channels, orig_h, orig_w = feature_map.shape

        new_h = int(math.ceil(orig_h / self.patch_h) * self.patch_h)
        new_w = int(math.ceil(orig_w / self.patch_w) * self.patch_w)

        interpolate = False
        if new_w != orig_w or new_h != orig_h:
            # Note: Padding can be done, but then it needs to be handled in attention function.
            feature_map = jax.image.resize(feature_map, shape=(in_channels, new_h, new_w), method="bilinear")
            interpolate = True

        # number of patches along width and height
        num_patch_w = new_w // patch_w # n_w
        num_patch_h = new_h // patch_h # n_h
        num_patches = num_patch_h * num_patch_w # N

        # [B, C, H, W] --> [B * C * n_h, p_h, n_w, p_w]
        reshaped_fm = jnp.reshape(feature_map,(in_channels * num_patch_h, patch_h, num_patch_w, patch_w))
        # [B * C * n_h, p_h, n_w, p_w] --> [B * C * n_h, n_w, p_h, p_w]
        transposed_fm = jnp.transpose(reshaped_fm, (0,2,1,3))
        # [B * C * n_h, n_w, p_h, p_w] --> [B, C, N, P] where P = p_h * p_w and N = n_h * n_w
        reshaped_fm = jnp.reshape(transposed_fm,(in_channels, num_patches, patch_area))
        # [B, C, N, P] --> [B, P, N, C]
        transposed_fm = jnp.transpose(reshaped_fm, (2,1,0))
        # [B, P, N, C] --> [BP, N, C]
        patches = transposed_fm#.reshape(patch_area, num_patches, -1)

        info_dict = {
            "orig_size": (orig_h, orig_w),
            "interpolate": interpolate,
            "total_patches": num_patches,
            "num_patches_w": num_patch_w,
            "num_patches_h": num_patch_h
        }

        return patches, info_dict

    def folding(self, patches, info_dict: Dict):
        # n_dim = patches.dim()
        assert len(patches.shape) == 3, "Tensor should be of shape BPxNxC. Got: {}".format(
            patches.shape
        )
        # [BP, N, C] --> [B, P, N, C]
        # patches = patches.contiguous().view(
        #     info_dict["batch_size"], self.patch_area, info_dict["total_patches"], -1
        # )

        pixels, num_patches, channels = patches.shape
        num_patch_h = info_dict["num_patches_h"]
        num_patch_w = info_dict["num_patches_w"]

        # [B, P, N, C] --> [B, C, N, P]
        # patches = patches.transpose(1, 3)
        patches = jnp.transpose(patches, (2,1,0))

        # [B, C, N, P] --> [B*C*n_h, n_w, p_h, p_w]
        feature_map = jnp.reshape(patches, (channels * num_patch_h, num_patch_w, self.patch_h, self.patch_w))
        # [B*C*n_h, n_w, p_h, p_w] --> [B*C*n_h, p_h, n_w, p_w]
        # feature_map = feature_map.transpose(1, 2)
        feature_map = jnp.transpose(feature_map, (0,2,1,3))
        # [B*C*n_h, p_h, n_w, p_w] --> [B, C, H, W]
        feature_map = jnp.reshape(feature_map, (channels, num_patch_h * self.patch_h, num_patch_w * self.patch_w))
        if info_dict["interpolate"]:
            feature_map = jax.image.resize(feature_map, (channels, info_dict["orig_size"][0], info_dict["orig_size"][1]), method="bilinear")
        return feature_map

    def __call__(self, x, state, key):
        res = x

        keys = jax.random.split(key, 3)
        # For MobileViTv3: Normal 3x3 convolution --> Depthwise 3x3 convolution
        fm_conv, state = self.local_rep(x, state)

        # convert feature map to patches
        patches, info_dict = self.unfolding(fm_conv)
        b_sz, n_patches, in_channels = patches.shape
        # learn global representations
        patches = self.global_rep(patches, key=keys[0])
        # print(patches.shape)
        patches = jnp.reshape(patches, (b_sz*n_patches, in_channels))
        # print((info_dict["num_patches_w"]*info_dict["num_patches_h"]*info_dict["total_patches"], -1))
        # print(patches.shape)
        # print(self.global_rep_out)
        patches = jax.vmap(self.global_rep_out)(patches)
        patches = jnp.reshape(patches, (b_sz, n_patches, in_channels))

        # [B x Patch x Patches x C] --> [B x C x Patches x Patch]
        fm = self.folding(patches=patches, info_dict=info_dict)
        # print("is oke")
        fm, state = self.conv_proj(fm, state)

        if self.fusion is not None:
            # For MobileViTv3: input+global --> local+global
            fm, state = self.fusion(
                jnp.concatenate((fm_conv, fm), 0), state
            )

        # For MobileViTv3: Skip connection
        fm = fm + res
        # print("is oke")
        return fm, state


        
class MobileViTv3(BaseEncoder):
    def __init__(self, opts, key, *args, **kwargs) -> None:
        num_classes = getattr(opts, "model.classification.n_classes", 1000)
        classifier_dropout = getattr(opts, "model.classification.classifier_dropout", 0.2)

        pool_type = getattr(opts, "model.layer.global_pool", "mean")
        image_channels = 3
        out_channels = 16

        mobilevit_config = get_configuration(opts=opts)

        # Segmentation architectures like Deeplab and PSPNet modifies the strides of the classification backbones
        # We allow that using `output_stride` arguments
        output_stride = kwargs.get("output_stride", None)
        dilate_l4 = dilate_l5 = False
        if output_stride == 8:
            dilate_l4 = True
            dilate_l5 = True
        elif output_stride == 16:
            dilate_l5 = True

        super(MobileViTv3, self).__init__()
        self.dilation = 1
        
        keys = jax.random.split(key, 7)
        
        self.model_conf_dict = dict()
        self.conv_1 = ConvLayer(
                opts=opts, in_channels=image_channels, out_channels=out_channels,
                kernel_size=3, stride=2, use_norm=True, use_act=True, key=keys[0]
            )
        
        self.model_conf_dict['conv1'] = {'in': image_channels, 'out': out_channels}

        in_channels = out_channels
        self.layer_1, out_channels = self._make_layer(
            opts=opts, input_channel=in_channels, cfg=mobilevit_config["layer1"], key=keys[1]
        )
        self.model_conf_dict['layer1'] = {'in': in_channels, 'out': out_channels}

        in_channels = out_channels
        self.layer_2, out_channels = self._make_layer(
            opts=opts, input_channel=in_channels, cfg=mobilevit_config["layer2"], key=keys[2]
        )
        self.model_conf_dict['layer2'] = {'in': in_channels, 'out': out_channels}
        
        in_channels = out_channels
        self.layer_3, out_channels = self._make_layer(
            opts=opts, input_channel=in_channels, cfg=mobilevit_config["layer3"], key=keys[3]
        )
        self.model_conf_dict['layer3'] = {'in': in_channels, 'out': out_channels}

        in_channels = out_channels
        self.layer_4, out_channels = self._make_layer(
            opts=opts, input_channel=in_channels, cfg=mobilevit_config["layer4"], dilate=dilate_l4, key=keys[4]
        )
        self.model_conf_dict['layer4'] = {'in': in_channels, 'out': out_channels}

        in_channels = out_channels
        self.layer_5, out_channels = self._make_layer(
            opts=opts, input_channel=in_channels, cfg=mobilevit_config["layer5"], dilate=dilate_l5, key=keys[4]
        )
        self.model_conf_dict['layer5'] = {'in': in_channels, 'out': out_channels}

        in_channels = out_channels
        exp_channels = min(mobilevit_config["last_layer_exp_factor"] * in_channels, 960)
        self.conv_1x1_exp = ConvLayer(
                opts=opts, in_channels=in_channels, out_channels=exp_channels,
                kernel_size=1, stride=1, use_act=True, use_norm=True, key=keys[6]
            )

        self.model_conf_dict['exp_before_cls'] = {'in': in_channels, 'out': exp_channels}

        classifier = []
        classifier.append(eqx.nn.Lambda(lambda x: jnp.mean(x, axis=(1, 2))))
        if 0.0 < classifier_dropout < 1.0:
            classifier.append(eqx.nn.Dropout(p=classifier_dropout))
        classifier.append(eqx.nn.Linear(in_features=exp_channels, out_features=num_classes, use_bias=True, key=keys[6])
        )
        self.classifier = eqx.nn.Sequential(classifier)
        # print(self.layer_1[0])
    
    def _make_layer(self, opts, input_channel, cfg: Dict, dilate: Optional[bool] = False, key=None):
        block_type = cfg.get("block_type", "mobilevit")
        if block_type.lower() == "mobilevit":
            return self._make_mit_layer(
                opts=opts,
                input_channel=input_channel,
                cfg=cfg,
                dilate=dilate,
                key=key,
            )
        else:
            return self._make_mobilenet_layer(
                opts=opts,
                input_channel=input_channel,
                cfg=cfg,
                key=key,
            )
    
    @staticmethod
    def _make_mobilenet_layer(opts, input_channel: int, cfg: Dict, key) -> Tuple[eqx.nn.Sequential, int]:
        output_channels = cfg.get("out_channels")
        num_blocks = cfg.get("num_blocks", 2)
        expand_ratio = cfg.get("expand_ratio", 4)
        block = []

        keys = jax.random.split(key, num_blocks)
        # print(num_blocks)
        for i in range(num_blocks):
            stride = cfg.get("stride", 1) if i == 0 else 1

            layer = InvertedResidual(
                opts=opts,
                in_channels=input_channel,
                out_channels=output_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                key = keys[i],
            )
            block.append(layer)
            input_channel = output_channels
        return eqx.nn.Sequential(block), input_channel

    def _make_mit_layer(self, opts, input_channel, cfg: Dict, dilate: Optional[bool] = False, key=None) -> Tuple[eqx.nn.Sequential, int]:
        prev_dilation = self.dilation
        block = []
        stride = cfg.get("stride", 1)

        keys = jax.random.split(key, 2)
        if stride == 2:
            if dilate:
                self.dilation *= 2
                stride = 1

            layer = InvertedResidual(
                opts=opts,
                in_channels=input_channel,
                out_channels=cfg.get("out_channels"),
                stride=stride,
                expand_ratio=cfg.get("mv_expand_ratio", 4),
                dilation=prev_dilation,
                key=keys[0],
            )

            block.append(layer)
            input_channel = cfg.get("out_channels")

        head_dim = cfg.get("head_dim", 32)
        transformer_dim = cfg["transformer_channels"]
        ffn_dim = cfg.get("ffn_dim")
        if head_dim is None:
            num_heads = cfg.get("num_heads", 4)
            if num_heads is None:
                num_heads = 4
            head_dim = transformer_dim // num_heads

        if transformer_dim % head_dim != 0:
            print("Transformer input dimension should be divisible by head dimension. "
                         "Got {} and {}.".format(transformer_dim, head_dim))
            assert False

        block.append(
            MobileViTv3Block(
                opts=opts,
                in_channels=input_channel,
                transformer_dim=transformer_dim,
                ffn_dim=ffn_dim,
                n_transformer_blocks=cfg.get("transformer_blocks", 1),
                patch_h=cfg.get("patch_h", 2),
                patch_w=cfg.get("patch_w", 2),
                dropout=getattr(opts, "model.classification.mit.dropout", 0.1),
                ffn_dropout=getattr(opts, "model.classification.mit.ffn_dropout", 0.0),
                attn_dropout=getattr(opts, "model.classification.mit.attn_dropout", 0.1),
                head_dim=head_dim,
                no_fusion=getattr(opts, "model.classification.mit.no_fuse_local_global_features", False),
                conv_ksize=getattr(opts, "model.classification.mit.conv_kernel_size", 3),
                key=keys[1],
            )
        )

        return eqx.nn.Sequential(block), input_channel
        
    def __call__(self, x, state, key):
        # print(self.layer_1)
        keys = jax.random.split(key, 8)
        x, state = self.conv_1(x, state, key=keys[0])
        x, state = self.layer_1(x, state=state, key=keys[1])
        x, state = self.layer_2(x, state=state, key=keys[2])
        
        x, state = self.layer_3(x, state=state, key=keys[3])
        x, state = self.layer_4(x, state=state, key=keys[4])
        x, state = self.layer_5(x, state=state, key=keys[5])
        x, state = self.conv_1x1_exp(x, state=state, key=keys[6])
        
        x = self.classifier(x, key=keys[7])
    
        print(x.shape)
        print(state)
        return x, state
        
def mobievit_xx_small_v3(key, n_classes=1000):
    import argparse
    opts = argparse.Namespace()
    
    setattr(opts,"model.classification.name","mobilevit_v3")
    setattr(opts,"model.classification.classifier_dropout", 0.1)

    setattr(opts,"model.classification.mit.mode" ,"xx_small_v3")
    setattr(opts,"model.classification.mit.ffn_dropout", 0.0)
    setattr(opts,"model.classification.mit.attn_dropout", 0.0)
    setattr(opts,"model.classification.mit.dropout", 0.05)
    setattr(opts,"model.classification.mit.number_heads", 4)
    setattr(opts,"model.classification.mit.no_fuse_local_global_features", False)
    setattr(opts,"model.classification.mit.conv_kernel_size", 3)

    setattr(opts,"model.classification.activation.name", "swish")

    setattr(opts,"model.normalization.name", "batch_norm_2d")
    setattr(opts,"model.normalization.momentum", 0.1)

    setattr(opts,"model.activation.name", "swish")

    setattr(opts,"model.activation.layer.global_pool", "mean")
    setattr(opts,"model.activation.layer.conv_init", "kaiming_normal")
    setattr(opts,"model.activation.layer.linear_init", "trunc_normal")
    setattr(opts,"model.activation.layer.linear_init_std_dev", 0.02)
    
    setattr(opts,"model.classification.n_classes", n_classes)

    # key = jax.random.PRNGKey(0)
    # x = jnp.ones((2,3,256,256))
    model =  MobileViTv3(opts, key)
    return model
    

if __name__ == "__main__":
    opts = argparse.Namespace()
    import argparse
    setattr(opts,"model.classification.name","mobilevit_v3")
    setattr(opts,"model.classification.classifier_dropout", 0.1)

    setattr(opts,"model.classification.mit.mode" ,"xx_small_v3")
    setattr(opts,"model.classification.mit.ffn_dropout", 0.0)
    setattr(opts,"model.classification.mit.attn_dropout", 0.0)
    setattr(opts,"model.classification.mit.dropout", 0.05)
    setattr(opts,"model.classification.mit.number_heads", 4)
    setattr(opts,"model.classification.mit.no_fuse_local_global_features", False)
    setattr(opts,"model.classification.mit.conv_kernel_size", 3)

    setattr(opts,"model.classification.activation.name", "swish")

    setattr(opts,"model.normalization.name", "batch_norm_2d")
    setattr(opts,"model.normalization.momentum", 0.1)

    setattr(opts,"model.activation.name", "swish")

    setattr(opts,"model.activation.layer.global_pool", "mean")
    setattr(opts,"model.activation.layer.conv_init", "kaiming_normal")
    setattr(opts,"model.activation.layer.linear_init", "trunc_normal")
    setattr(opts,"model.activation.layer.linear_init_std_dev", 0.02)
    
    setattr(opts,"model.classification.n_classes", 1000)

    key = jax.random.PRNGKey(0)
    x = jnp.ones((2,3,256,256))
    model =  MobileViTv3(opts, key)
    state = eqx.nn.State(model)
    
    import optax 
    
    optim = optax.adamw(0.0001)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    
    @eqx.filter_jit
    @eqx.filter_grad
    def loss(model, x, y):
        pred_y, _ = jax.vmap(model, in_axes=(0,None,None), out_axes=(0), axis_name="batch")(x, state, key)
        print(pred_y.shape)
        return optax.softmax_cross_entropy_with_integer_labels(pred_y, y).mean()

    
    grads = loss(model, x, jnp.array([9,5]))
    
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    # print(len(grads))
    # learning_rate = 0.1
    # a = jax.tree_util.tree_map(lambda m, g: None if g==None else m.shape, model, grads)
    # print(a)
    # new_model = jax.tree_util.tree_map(lambda m, g: m - learning_rate * g, model, grads)