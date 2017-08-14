import unittest
from firedrake import *
import numpy as np

from pySDC.implementations.datatype_classes.firedrake_mesh import mesh, rhs_imex_mesh

class TestFiredrake(unittest.TestCase):

 
  #
  # General setUp function used by all tests
  #
  def setUp(self):
    pass

  # ***************
  # **** TESTS ****
  # ***************

  #
  # Check that a mesh object can be instantiated given init and val
  #
  def test_caninstantiate(self):
    fd = mesh(init=16, val=0.0)

  #
  # Check that copy constructor works
  #
  def test_caninstantiate_copy(self):
    fd = mesh(init=16, val=-1.0)
    fd2 = mesh(fd)

  def test_canabs(self): 
    fd = mesh(init=16, val=-1.0)
    assert abs(fd)==1.0, "Function abs of firedrake_mesh returned wrong value."

  def test_canadd(self):
    fd1 = mesh(init=16, val=1.0)
    fd2 = mesh(init=16, val=-2.0)
    fd3 = fd1 + fd2
    print(isinstance(fd3, mesh))
    assert abs(fd3)==1.0, "After addition, function abs returned wrong value."

  def test_cansub(self):
    fd1 = mesh(init=16, val=1.0)
    fd2 = mesh(init=16, val=-2.0)
    fd3 = fd1 - fd2
    print(isinstance(fd3, mesh))
    assert abs(fd3)==3.0, "After subtraction, function abs returned wrong value."

  def test_canmult(self):
    fd1 = mesh(init=16, val=-2.0)
    a   = 2.0
    fd3 = a * fd1
    print(isinstance(fd3, mesh))
    assert abs(fd3)==4.0, "After multiplication, function abs returned wrong value."

  def test_caninstantiate_imex(self):
    fd = rhs_imex_mesh(init=16)


