from abc import ABCMeta, abstractmethod
import math


class AbstractPrimitive(metaclass=ABCMeta):
    """
    Use this class when creating new operations for edges.

    This is required because we are agnostic to operations
    at the edges. As a consequence, they can contain subgraphs
    which requires naslib to detect and properly process them.
    """

    def __init__(self, kwargs):
        super().__init__()

        self.init_params = {
            k: v
            for k, v in kwargs.items()
            if k != "self" and not k.startswith("_") and k != "kwargs"
        }

    @abstractmethod
    def get_embedded_ops(self):
        """
        Return any embedded ops so that they can be
        analysed whether they contain a child graph, e.g.
        a 'motif' in the hierachical search space.

        If there are no embedded ops, then simply return
        `None`. Should return a list otherwise.
        """
        raise NotImplementedError()

    @property
    def get_op_name(self):
        return type(self).__name__


class AbstractCombOp(metaclass=ABCMeta):
    """
    Use this class to create custom combination operations to be used in nodes.
    """

    def __init__(self, comb_op):
        self.comb_op = comb_op

    @property
    def op_name(self):
        return type(self).__name__


class EdgeNormalizationCombOp(AbstractCombOp):
    """
    Combination operation to use for edge normalization.

    Returns the weighted sum of input tensors based on the (softmax of) edge weights.
    """


class MixedOp(AbstractPrimitive):
    """
    Continous relaxation of the discrete search space.
    """

    def __init__(self, primitives):
        super().__init__(locals())
        self.primitives = primitives
        self._add_primitive_modules()
        self.pre_process_hook = None
        self.post_process_hook = None

    def _add_primitive_modules(self):
        for i, primitive in enumerate(self.primitives):
            self.add_module("primitive-{}".format(i), primitive)

    def set_pre_process_hook(self, fn):
        self.set_pre_process_hook = fn

    def set_post_process_hook(self, fn):
        self.post_process_hook = fn

    @abstractmethod
    def get_weights(self, edge_data):
        raise NotImplementedError()

    @abstractmethod
    def process_weights(self, weights):
        raise NotImplementedError()

    @abstractmethod
    def apply_weights(self, x, weights):
        raise NotImplementedError()

    def get_embedded_ops(self):
        return self.primitives

    def set_embedded_ops(self, primitives):
        self.primitives = primitives
        self._add_primitive_modules()


class PartialConnectionOp(AbstractPrimitive):
    """
    Partial Connection Operation.

    This class takes a MixedOp and replaces its primitives with the fewer channel version of those primitives.
    """

    def __init__(self, mixed_op: MixedOp, k: int):
        super().__init__(locals())
        self.k = k
        self.mixed_op = mixed_op

        pc_primitives = []
        for primitive in mixed_op.get_embedded_ops():
            pc_primitives.append(self._create_pc_primitive(primitive))

        self.mixed_op.set_embedded_ops(pc_primitives)

    def _create_pc_primitive(self, primitive: AbstractPrimitive) -> AbstractPrimitive:
        """
        Creates primitives with fewer channels for Partial Connection operation.
        """
        init_params = primitive.init_params

        try:
            # TODO: Force all AbstractPrimitives with convolutions to use 'C_in' and 'C_out' in the initializer
            init_params["C_in"] = init_params["C_in"] // self.k

            if "C_out" in init_params:
                init_params["C_out"] = init_params["C_out"] // self.k
            elif "C" in init_params:
                init_params["C"] = init_params["C"] // self.k
        except KeyError:
            return primitive

        pc_primitive = primitive.__class__(**init_params)
        return pc_primitive


    def get_embedded_ops(self):
        return self.mixed_op.get_embedded_ops()


class Identity(AbstractPrimitive):
    """
    An implementation of the Identity operation.
    """

    def __init__(self, **kwargs):
        super().__init__(locals())

    def forward(self, x, edge_data=None):
        return x

    def get_embedded_ops(self):
        return None


class Zero(AbstractPrimitive):
    """
    Implementation of the zero operation. It removes
    the connection by multiplying its input with zero.
    """

    def __init__(self, stride, C_in=None, C_out=None, **kwargs):
        """
        When setting stride > 1 then it is assumed that the
        channels must be doubled.
        """
        super().__init__(locals())
        self.stride = stride
        self.C_in = C_in
        self.C_out = C_out


    def get_embedded_ops(self):
        return None

    def __repr__(self):
        return "Zero (stride={})".format(self.stride)


class Zero1x1(AbstractPrimitive):
    """
    Implementation of the zero operation. It removes
    the connection by multiplying its input with zero.
    """

    def __init__(self, stride, **kwargs):
        """
        When setting stride > 1 then it is assumed that the
        channels must be doubled.
        """
        super().__init__(locals())
        self.stride = stride

    def get_embedded_ops(self):
        return None

    def __repr__(self):
        return "Zero1x1 (stride={})".format(self.stride)


class SepConv(AbstractPrimitive):
    """
    Implementation of Separable convolution operation as
    in the DARTS paper, i.e. 2 sepconv directly after another.
    """

    def __init__(
        self, C_in, C_out, kernel_size, stride, padding, affine=True, **kwargs
    ):
        super().__init__(locals())
        self.kernel_size = kernel_size

    def get_embedded_ops(self):
        return None

    @property
    def get_op_name(self):
        op_name = super().get_op_name
        op_name += "{}x{}".format(self.kernel_size, self.kernel_size)
        return op_name


class DilConv(AbstractPrimitive):
    """
    Implementation of a dilated separable convolution as
    used in the DARTS paper.
    """

    def __init__(
        self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True, **kwargs
    ):
        super().__init__(locals())
        self.kernel_size = kernel_size

    def get_embedded_ops(self):
        return None

    @property
    def get_op_name(self):
        op_name = super().get_op_name
        op_name += "{}x{}".format(self.kernel_size, self.kernel_size)
        return op_name


class Stem(AbstractPrimitive):
    """
    This is used as an initial layer directly after the
    image input.
    """

    def __init__(self, C_in=3, C_out=64, **kwargs):
        super().__init__(locals())

    def get_embedded_ops(self):
        return None


class Sequential(AbstractPrimitive):
    """
    Implementation of `torch.nn.Sequential` to be used
    as op on edges.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(locals())
        self.primitives = args

    def get_embedded_ops(self):
        return list(self.primitives)


class MaxPool(AbstractPrimitive):
    def __init__(self, C_in, kernel_size, stride, use_bn=True, **kwargs):
        super().__init__(locals())
        self.kernel_size = kernel_size

    def forward(self, x, edge_data):
        x = self.maxpool(x)
        return x

    def get_embedded_ops(self):
        return None


class MaxPool1x1(AbstractPrimitive):
    """
    Implementation of MaxPool with an optional 1x1 convolution
    in case stride > 1. The 1x1 convolution is required to increase
    the number of channels.
    """

    def __init__(
        self, kernel_size, stride, C_in=None, C_out=None, affine=True, **kwargs
    ):
        super().__init__(locals())
        self.stride = stride

    def get_embedded_ops(self):
        return None


class AvgPool(AbstractPrimitive):
    """
    Implementation of Avergae Pooling.
    """

    def __init__(self, C_in, kernel_size, stride, use_bn=True, **kwargs):
        super().__init__(locals())

    def get_embedded_ops(self):
        return None


class AvgPool1x1(AbstractPrimitive):
    """
    Implementation of Avergae Pooling with an optional
    1x1 convolution afterwards. The convolution is required
    to increase the number of channels if stride > 1.
    """

    def __init__(
        self, kernel_size, stride, C_in=None, C_out=None, affine=True, **kwargs
    ):
        super().__init__(locals())
        self.stride = stride
        if stride > 1:
            assert C_in is not None and C_out is not None
            self.affine = affine
            self.C_in = C_in
            self.C_out = C_out

    def get_embedded_ops(self):
        return None


class GlobalAveragePooling(AbstractPrimitive):
    """
    Just a wrapper class for averaging the input across the height and width dimensions
    """

    def __init__(self):
        super().__init__(locals())

    def get_embedded_ops(self):
        return None


class ReLUConvBN(AbstractPrimitive):
    """
    Implementation of ReLU activation, followed by 2d convolution and then 2d batch normalization.
    """

    def __init__(
        self,
        C_in,
        C_out,
        kernel_size,
        stride=1,
        affine=True,
        bias=False,
        track_running_stats=True,
        **kwargs,
    ):
        super().__init__(locals())
        self.kernel_size = kernel_size
        pad = 0 if kernel_size == 1 else 1

    def get_embedded_ops(self):
        return None

    @property
    def get_op_name(self):
        op_name = super().get_op_name
        op_name += "{}x{}".format(self.kernel_size, self.kernel_size)
        return op_name


class ConvBnReLU(AbstractPrimitive):
    """
    Implementation of 2d convolution, followed by 2d batch normalization and ReLU activation.
    """

    def __init__(self, C_in, C_out, kernel_size, stride=1, affine=True, **kwargs):
        super().__init__(locals())
        self.kernel_size = kernel_size
        pad = 0 if stride == 1 and kernel_size == 1 else 1

    def get_embedded_ops(self):
        return None

    @property
    def get_op_name(self):
        op_name = super().get_op_name
        op_name += "{}x{}".format(self.kernel_size, self.kernel_size)
        return op_name


class InputProjection(AbstractPrimitive):
    """
    Implementation of a 1x1 projection, followed by an abstract primitive model.
    """

    def __init__(self, C_in: int, C_out: int, primitive: AbstractPrimitive):
        """
        Args:
            C_in        : Number of input channels
            C_out       : Number of output channels
            primitive   : Module of AbstractPrimitive type to which the projected input will be fed
        """
        super().__init__(locals())
        self.module = primitive

    def get_embedded_ops(self):
        return None

    @property
    def get_op_name(self):
        op_name = super().get_op_name
        op_name += f"{self.module.get_op_name()}"
        return op_name


class Concat1x1:
    # TODO Figure out how this thing gets its name
    """
    Implementation of the channel-wise concatination followed by a 1x1 convolution
    to retain the channel dimension.
    """


class StemJigsaw(AbstractPrimitive):
    """
    This is used as an initial layer directly after the
    image input.
    """

    def get_embedded_ops(self):
        return None


class SequentialJigsaw(AbstractPrimitive):
    """
    Implementation of `torch.nn.Sequential` to be used
    as op on edges.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(locals())
        self.primitives = args

    def get_embedded_ops(self):
        return list(self.primitives)


class GenerativeDecoder(AbstractPrimitive):
    def __init__(self, in_dim, target_dim, target_num_channel=3, norm=None):
        super(GenerativeDecoder, self).__init__(locals())

        in_channel, in_width = in_dim[0], in_dim[1]
        out_width = target_dim[0]
        num_upsample = int(math.log2(out_width / in_width))
        assert num_upsample in [2, 3, 4, 5, 6], f"invalid num_upsample: {num_upsample}"

    def get_embedded_ops(self):
        return None
