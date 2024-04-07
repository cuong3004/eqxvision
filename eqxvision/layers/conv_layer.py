import equinox as eqx
from typing import Optional, Union, Tuple
import jax 

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
        
class ConvLayer(eqx.nn.Sequential):
    def __init__(
        self,
        opts,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = 1,
        dilation: Optional[Union[int, Tuple[int, int]]] = 1,
        groups: Optional[int] = 1,
        bias: Optional[bool] = False,
        padding_mode: Optional[str] = "zeros",
        use_norm: Optional[bool] = True,
        use_act: Optional[bool] = True,
        act_name: Optional[str] = None,
        key = None,
        *args,
        **kwargs
    ) -> None:
        if use_norm:
            norm_type = getattr(opts, "model.normalization.name", "batch_norm")
            if norm_type is not None and norm_type.find("batch") > -1:
                assert not bias, "Do not use bias when using normalization layers."
            elif norm_type is not None and norm_type.find("layer") > -1:
                bias = True
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        if isinstance(stride, int):
            stride = (stride, stride)

        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        assert isinstance(kernel_size, Tuple)
        assert isinstance(stride, Tuple)
        assert isinstance(dilation, Tuple)

        padding = (
            int((kernel_size[0] - 1) / 2) * dilation[0],
            int((kernel_size[1] - 1) / 2) * dilation[1],
        )

        if in_channels % groups != 0:
            print(
                "Input channels are not divisible by groups. {}%{} != 0 ".format(
                    in_channels, groups
                )
            )
        if out_channels % groups != 0:
            print(
                "Output channels are not divisible by groups. {}%{} != 0 ".format(
                    out_channels, groups
                )
            )

        block = []#eqx.nn.Sequential()
        
        keys = jax.random.split(key, 3)

        conv_layer = eqx.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            use_bias=bias,
            # padding_mode=padding_mode,
            key = keys[0]
        )

        block.append(conv_layer)

        # self.norm_name = None
        if use_norm:
            norm_layer = get_normalization_layer(opts=opts, num_features=out_channels)
            block.append(norm_layer)
            # self.norm_name = norm_layer.__class__.__name__

        # self.act_name = None
        act_type = (
            getattr(opts, "model.activation.name", "prelu")
            if act_name is None
            else act_name
        )

        if act_type is not None and use_act:
            neg_slope = getattr(opts, "model.activation.neg_slope", 0.1)
            inplace = getattr(opts, "model.activation.inplace", False)
            act_layer = get_activation_fn(
                act_type=act_type,
                inplace=inplace,
                negative_slope=neg_slope,
                num_parameters=out_channels,
            )
            block.append(act_layer)
            # self.act_name = act_layer.__class__.__name__
        
        self.layers = eqx.nn.Sequential(block)
        # self.in_channels = in_channels
        # self.out_channels = out_channels
        # self.stride = stride
        # self.groups = groups
        # self.kernel_size = conv_layer.kernel_size
        # self.bias = bias
        # self.dilation = dilation

        
