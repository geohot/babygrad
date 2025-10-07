from __future__ import annotations
import functools, operator, weakref
from typing import Any, Literal, Final, Iterable, TypeVar, Sequence
from enum import auto, Enum
from dataclasses import dataclass

# *** helpers ***

FmtStr = Literal['?', 'b', 'B', 'h', 'H', 'i', 'I', 'q', 'Q', 'e', 'f', 'd']
ConstType = float|int|bool

T = TypeVar("T")
def prod(x:Iterable[T]) -> T|int: return functools.reduce(operator.mul, x, 1)
def assert_all_same(items):
  assert all(x == items[0] for x in items), f"mismatch in {items}"
  return items[0]
def argfix(*x):
  if x and x[0].__class__ in (tuple, list):
    if len(x) != 1: raise ValueError(f"bad arg {x}")
    return tuple(x[0])
  return x
def make_tuple(x:int|Sequence[int], cnt:int) -> tuple[int, ...]: return (x,)*cnt if isinstance(x, int) else tuple(x)

# *** ops ***

# TODO: should this include MULTI? probably
class AddrSpace(Enum): GLOBAL = auto(); LOCAL = auto(); REG = auto()  # noqa: E702

# TODO: the type of arg should depend on op, is this doable?
class Ops(Enum):
  # hmm, i don't like DTYPE as an Op. it has similar vibes to DEVICE
  DTYPE = auto()
  DEVICE = auto()

  # a CONST has a value, a DTYPE, and an optional DEVICE
  CONST = auto()
  BUFFER = auto() # <-- all types?

  RANGE = auto()

  # unary ops
  CAST = auto(); BITCAST = auto() # noqa: E702
  EXP2 = auto(); LOG2 = auto(); SIN = auto() # noqa: E702
  SQRT = auto(); RECIP = auto(); NEG = auto(); TRUNC = auto() # noqa: E702

  # BinaryOps
  ADD = auto(); MUL = auto(); POW = auto(); IDIV = auto(); MAX = auto() # noqa: E702
  CMPLT = auto(); CMPNE = auto() # noqa: E702

  # TernaryOps
  WHERE = auto()

  # reduce axis -> reduce -> store+load
  REDUCE_AXIS = auto(); REDUCE = auto() # noqa: E702

  STORE = auto(); LOAD = auto()

  # movement ops!
  RESHAPE = auto(); EXPAND = auto() # noqa: E702
  SHRINK = auto(); PAD = auto() # noqa: E702
  PERMUTE = auto(); FLIP = auto() # noqa: E702

class GroupOp:
  Unary = {Ops.CAST, Ops.BITCAST, Ops.EXP2, Ops.LOG2, Ops.SIN, Ops.SQRT, Ops.RECIP, Ops.NEG, Ops.TRUNC}
  Binary = {Ops.ADD, Ops.MUL, Ops.POW, Ops.IDIV, Ops.MAX, Ops.CMPLT, Ops.CMPNE}
  Movement = {Ops.RESHAPE, Ops.EXPAND, Ops.PERMUTE, Ops.PAD, Ops.SHRINK, Ops.FLIP}

class UOpMetaClass(type):
  # dedup all UOps
  ucache:dict[tuple, weakref.ReferenceType[UOp]] = {}
  def __call__(cls, op:Ops, *src:UOp, arg:Any=None):
    if (wret:=UOpMetaClass.ucache.get(key:=(op, src, arg), None)) is not None and (ret:=wret()) is not None: return ret
    UOpMetaClass.ucache[key] = weakref.ref(created:=super().__call__(op, *src, arg=arg))
    return created

class UOp(metaclass=UOpMetaClass):
  # TODO: tinygrad -- can we change the UOp constructor to this?
  def __init__(self, op:Ops, *src:UOp, arg:Any=None): self.op, self.src, self.arg = op, src, arg
  def __repr__(self): return f"UOp({", ".join([str(self.op)]+[str(x) for x in self.src])}" + (f", arg={self.arg})" if self.arg is not None else ")")

  # constructed properties

  @functools.cached_property
  def dtype(self) -> UOp:
    # TODO: tinygrad -- dtype should be a constructed property
    if self.op is Ops.DTYPE: return self
    if self.op in GroupOp.Movement: return self.src[0].dtype
    return assert_all_same([x.dtype for x in self.src])

  @functools.cached_property
  def shape(self) -> list[UOp]|None:
    if self.op is Ops.DTYPE: return None
    if self.op is Ops.CONST: return []
    # TODO: tinygrad -- RESHAPE/EXPAND/SHRINK/PAD should have arguments as UOp srcs
    if self.op is Ops.RESHAPE:
      #assert prod(self.src[0].shape) == prod(self.src[1:]), "reshape must preserve shape"
      return self.src[1:]
    if self.op is Ops.EXPAND:
      #assert all(s1 == s2 or s1 == 1 for s1,s2 in zip(self.src[0].shape, self.src[1:])), "expand only expands 1s"
      return self.src[1:]
    return assert_all_same([x.shape for x in self.src])

def smax(*lst):
  # TODO: write this
  return sorted(argfix(*lst), key=lambda x: x.arg)[-1]

# *** high level ***

sint = int|UOp

@dataclass(frozen=True, eq=False, slots=True)
class DType:
  # TODO: tinygrad -- do we need priority?
  itemsize: int
  name: str
  fmt: FmtStr|None
  def __repr__(self): return f"dtypes.{self.name}"

class dtypes:
  # TODO: tinygrad -- these should be UOps to not repeat the deduping logic
  index: Final[UOp] = UOp(Ops.DTYPE, arg=DType(0, "index", None))
  bool: Final[UOp] = UOp(Ops.DTYPE, arg=DType(1, "bool", '?'))
  int: Final[UOp] = UOp(Ops.DTYPE, arg=DType(4, "int", 'i'))
  float: Final[UOp] = UOp(Ops.DTYPE, arg=DType(4, "float", 'f'))

def py_to_dtype(data) -> UOp:
  if isinstance(data, float): return dtypes.float
  if isinstance(data, int): return dtypes.int
  if isinstance(data, bool): return dtypes.bool
  raise RuntimeError("unsupported data")

def fix_shape(shape:tuple[sint, ...]) -> list[UOp]:
  return [UOp(Ops.CONST, dtypes.index, arg=s) if isinstance(s, int) else s for s in shape]

# *** Tensor helpers ***

def _align_left(*shapes:tuple[sint, ...]) -> tuple[tuple[sint, ...], ...]:
  # unsqueeze left to make every shape same length
  max_dim = max(len(shape) for shape in shapes)
  return tuple((1,) * (max_dim - len(shape)) + shape for shape in shapes)
def _broadcast_shape(*shapes:tuple[sint, ...]) -> tuple[sint, ...]:
  return tuple(0 if 0 in nth_dim_sizes else smax(nth_dim_sizes) for nth_dim_sizes in zip(*_align_left(*shapes)))

# *** Tensor ***

class Tensor:
  def __init__(self, data:float|int|bool|UOp):
    if isinstance(data, UOp):
      self.uop = data
    else:
      # const
      self.uop = UOp(Ops.CONST, py_to_dtype(data), arg=data)
    # do construction early to find errors
    self.dtype, self.shape

  @property
  def dtype(self): return self.uop.dtype
  @property
  def shape(self): return self.uop.shape
  @property
  def ndim(self): return len(self.shape)

  def _resolve_dim(self, dim:int) -> int:
    total = self.ndim
    if not -max(1, total) <= dim <= max(1, total)-1: raise IndexError(f"{dim=} out of range {[-max(1, total), max(1, total)-1]}")
    return dim + total if dim < 0 else dim

  def __repr__(self): return repr(self.uop)

  def __mul__(self, x:Tensor) -> Tensor:
    out_shape = _broadcast_shape(self.shape, x.shape)
    # TODO: broadcasting + constcasting
    return Tensor(UOp(Ops.MUL, self.expand(out_shape).uop, x.expand(out_shape).uop))

  def reshape(self, *shape:sint) -> Tensor: return Tensor(UOp(Ops.RESHAPE, self.uop, *fix_shape(argfix(*shape))))
  def expand(self, *shape:sint) -> Tensor: return Tensor(UOp(Ops.EXPAND, self.uop, *fix_shape(argfix(*shape))))
  def permute(self, order:tuple[int, ...]) -> Tensor:
    order_arg = tuple(self._resolve_dim(x) for x in argfix(order))
    if sorted(order_arg) != list(range(self.ndim)): raise RuntimeError(f"order is not a valid permutation, getting {order_arg}")
    return Tensor(UOp(Ops.PERMUTE, self.uop, arg=order_arg))

  def transpose(self, dim0=1, dim1=0) -> Tensor:
    order = list(range(self.ndim))
    order[dim0], order[dim1] = order[dim1], order[dim0]
    return self.permute(order)

  @staticmethod
  def full(shape:tuple[sint, ...], fill_value:ConstType, **kwargs) -> Tensor:
    return Tensor(fill_value).reshape((1, )*len(new_shape := argfix(shape))).expand(new_shape)
  @staticmethod
  def ones(*shape, **kwargs) -> Tensor:
    return Tensor.full(argfix(*shape), 1.0, **kwargs)

  def _reduce(self, op:Ops, axis:int|Sequence[int]|None=None, keepdim=False) -> Tensor:
    axis = tuple(self._resolve_dim(x) for x in (range(self.ndim) if axis is None else make_tuple(axis, 1)))
    if self.ndim == 0: axis = ()
    ret = Tensor(UOp(Ops.REDUCE_AXIS, self.uop, arg=axis))
    return ret if keepdim else ret.reshape(tuple(s for i,s in enumerate(self.shape) if i not in axis))

  def sum(self, axis:int|Sequence[int]|None=None, keepdim=False) -> Tensor:
    return self._reduce(Ops.ADD, axis, keepdim)

  def __matmul__(self, w:Tensor) -> Tensor:
    x, dx, dw = self, self.ndim, w.ndim
    if not (dx > 0 and dw > 0): raise RuntimeError(f"both tensors need to be at least 1D, got {dx}D and {dw}D")
    if x.shape[-1] != w.shape[(axis_w:=-min(w.ndim,2))]: raise RuntimeError(f"cannot dot {x.shape} and {w.shape}")
    x = x.reshape(*x.shape[0:-1], *[1]*min(dx-1, dw-1, 1), x.shape[-1])
    w = w.reshape(*w.shape[0:-2], *[1]*min(dx-1, dw-1, 1), *w.shape[axis_w:]).transpose(-1, axis_w)
    return (x*w).sum(-1)
