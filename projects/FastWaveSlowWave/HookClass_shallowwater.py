from __future__ import division
from pySDC.core.Hooks import hooks
from firedrake import File

class dump_solution(hooks):

    def __init__(self):
      super(dump_solution, self).__init__()
      self.output = File("shallow_water.pvd")

    def pre_run(self, step, level_number):
      super(dump_solution, self).post_step(step, level_number)
      L = step.levels[level_number]
      self.output.write(L.u[0].f)

    def post_step(self, step, level_number):
      super(dump_solution, self).post_step(step, level_number)
      L = step.levels[level_number]
      self.output.write(L.uend.f)
