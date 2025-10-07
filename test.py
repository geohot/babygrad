import unittest
from lib import UOp, Ops, Tensor

class TestUOp(unittest.TestCase):
  def test_dedup(self):
    a1 = UOp(Ops.CONST, arg=4)
    a2 = UOp(Ops.CONST, arg=4)
    assert a1 is a2

class TestTensor(unittest.TestCase):
  def test_mul(self):
    out = Tensor(2) * Tensor(2)
    print(out)

  def test_full(self):
    full = Tensor.ones(8,8)
    print(full)

  def test_gemm(self):
    t1 = Tensor.ones(8,8)
    t2 = Tensor.ones(8,8)
    out = t1 @ t2
    print(out)

if __name__ == "__main__":
  unittest.main()
