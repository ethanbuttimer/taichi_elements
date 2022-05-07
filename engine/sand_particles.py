from engine.mesh_io import write_point_cloud
import numpy as np
import taichi as ti
import time
import gc

# all defined in particle creation, update this later
PART_VOLUME = 1 # set to arbitrary value -- not sure what this should be
LAMBDA = 1
MU = 1

class SandParticle:
    def __init__(self):
        self.temp = 1 # set to arbitrary value -- not sure what this should be
    
    def derivative(self, s):
        s_ln = ti.Matrix.zero(ti.f32, 2, 2)
        s_inverse = ti.Matrix.zero(ti.f32, 2, 2)
        s_ln[0,0] = ti.log(s[0,0])
        s_ln[1,1] = ti.log(s[1,1])
        s_inverse[0,0] = 1.0 / s[0,0]
        s_inverse[1,1] = 1.0 / s[1,1]
        out = 2 * MU * s_inverse * s_ln + LAMBDA * s_ln.trace() * s_inverse
        return out

    def energy_derivative(self, elastic_deformation):
        # get svd of elasticity term 
        U, S, V = ti.svd(elastic_deformation)
        S_update = U.inverse() * elastic_deformation * V.transpose().inverse()
        T = self.derivative(S_update)
        A = U * T * V.transpose() * elastic_deformation.transpose()
        return PART_VOLUME * A
    
