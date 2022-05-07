from engine.mesh_io import write_point_cloud
import numpy as np
import taichi as ti
import time
import gc

# all defined in particle creation, update this later
PART_VOLUME = 1 # set to arbitrary value -- not sure what this should be
LAMBDA = 1
MU = 1
HARDNESS = 1
GAMMA = 1

class WaterParticle:
    def __init__(self):
        self.temp = 1 # set to arbitrary value -- not sure what this should be

    def energy_derivative(self, J):
        # Je = ti.determinant(elastic_deformation)
        P = HARDNESS * (1 / ti.power(J, GAMMA) - 1)
        temp = ti.Matrix.identity()
        out = -1 * P * temp
        return PART_VOLUME * out

    
