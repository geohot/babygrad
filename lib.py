from __future__ import annotations
import functools, operator, weakref
from typing import Any, Literal, Final, Iterable, TypeVar
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
  CAST = auto()

  # binary ops
  ADD = auto(); MUL = auto() # noqa: E702

  # reduce axis -> reduce -> store+load
  REDUCE_AXIS = auto(); REDUCE = auto() # noqa: E702

  STORE = auto(); LOAD = auto()

  # movement ops!
  RESHAPE = auto(); EXPAND = auto() # noqa: E702
  SHRINK = auto(); PAD = auto() # noqa: E702
  PERMUTE = auto(); FLIP = auto() # noqa: E702

class GroupOp:
  Movement = {Ops.RESHAPE, Ops.EXPAND, Ops.PERMUTE, Ops.PAD, Ops.SHRINK, Ops.FLIP}

class UOpMetaClass(type):
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

  def __repr__(self): return repr(self.uop)

  def __mul__(self, x:Tensor) -> Tensor:
    # TODO: broadcasting + constcasting
    return Tensor(UOp(Ops.MUL, self.uop, x.uop))

  def reshape(self, *shape:sint) -> Tensor: return Tensor(UOp(Ops.RESHAPE, self.uop, *fix_shape(argfix(*shape))))
  def expand(self, *shape:sint) -> Tensor: return Tensor(UOp(Ops.EXPAND, self.uop, *fix_shape(argfix(*shape))))

  @staticmethod
  def full(shape:tuple[sint, ...], fill_value:ConstType, **kwargs) -> Tensor:
    return Tensor(fill_value).reshape((1, )*len(new_shape := argfix(shape))).expand(new_shape)
  @staticmethod
  def ones(*shape, **kwargs) -> Tensor:
    return Tensor.full(argfix(*shape), 1.0, **kwargs)
