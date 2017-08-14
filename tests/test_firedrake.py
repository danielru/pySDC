import unittest

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
