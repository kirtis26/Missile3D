#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from easyvec import Vec3, Mat3
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ipywidgets import interact, FloatSlider
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform


# In[ ]:


class Target3D(object):
       
    @classmethod
    def get_simple_target(cls, pos, vel, aim, vel_aim, g=9.80665, dt=0.01, time_min=True):
        pos = cls._validate_dimension_argument(pos)
        vel = cls._validate_dimension_argument(vel)
        aim = cls._validate_dimension_argument(aim)
        vel_aim = cls._validate_dimension_argument(vel_aim)
        target = cls(pos=pos, vel=vel, aim=aim, vel_aim=vel_aim, g=g, dt=dt, time_min=time_min)
        parameters_simple_target = np.array([pos[0], pos[1], pos[2], vel[0], vel[1], vel[2], 0])
        target.set_init_cond(init_parametrs=parameters_simple_target)
        return target
    
    def __init__(self, **kwargs):
        self.g = kwargs['g']
        self.t = 0
        self.dt = kwargs['dt']
        
        self.A = kwargs['pos']
        self.D = kwargs['aim']
        self.velA = kwargs['vel'] 
        self.velD = kwargs['vel_aim']
        
        self.fly_time = 0
        self._set_fly_time(kwargs['time_min'])
        
        self.n = 7
        self.state = np.zeros(self.n)
        self.state_init = np.zeros(self.n)
    
    @property
    def pos(self):
        return self._fpos(self.t)
    
    @property
    def x(self):
        return self._fpos(self.t)[0]

    @property
    def y(self):
        return self._fpos(self.t)[1]
    
    @property
    def z(self):
        return self._fpos(self.t)[2]
    
    @property
    def vel(self):
        return self._fvel(self.t)
    
    @property
    def vel_abs(self):
        return np.sqrt(self.vel.dot(self.vel))
    
    @property
    def vel_x(self):
        return self.vel[0]
    
    @property
    def vel_y(self):
        return self.vel[1]
    
    @property
    def vel_z(self):
        return self.vel[2]
    
    def set_init_cond(self, init_parametrs=None):
        if init_parametrs is None:
            init_parametrs = self.get_random_parameters_of_target()
        self.state = np.array(init_parametrs)
        self.state_init = np.array(init_parametrs)

    def get_random_parameters_of_target(self):
        # TODO
        pass
    
    def reset(self):
        self.set_state(self.state_init)

    def set_state(self, state):
        self.state = np.array(state)

    def get_state(self):
        return self.state
    
    def get_state_init(self):
        return self.state_init
    
    def get_B(self, fly_time):
        return self.A + (self.velA * fly_time / 3)
    
    def get_C(self, fly_time): 
        return self.D - (self.velD * fly_time / 3)
    
    def get_traject(self, fly_time, vel_aim=None, n_points=100):
        vel_aim = self.velD if vel_aim == None else vel_aim
        taus = np.linspace(0, 1, n_points)
        A, velA = self.A, self.velA
        D, velD = self.D, vel_aim
        B = self.get_B(fly_time)
        C = self.get_C(fly_time)
        return np.array([
            (1-tau)**3 * A + 3*tau*(1-tau)**2 * B + 3*tau*tau*(1-tau)*C + tau**3 * D
            for tau in taus])
    
    def get_traject_vels(self, fly_time, vel_aim=None, n_points=100):
        vel_aim = self.velD if vel_aim == None else vel_aim
        taus = np.linspace(0, 1, n_points)
        A, velA = self.A, self.velA
        D, velD = self.D, vel_aim
        B = self.get_B(fly_time)
        C = self.get_C(fly_time)
        return np.array([
            (3*(1-tau)**2*(B-A) + 6*tau*(1-tau)*(C-B) + 3*tau**2*(D-C)) / fly_time
            for tau in taus
        ])
    
    def get_traject_acc(self, fly_time, vel_aim=None, n_points=100):
        vel_aim = self.velD if vel_aim == None else vel_aim
        taus = np.linspace(0, 1, n_points)
        A, velA = self.A, self.velA
        D, velD = self.D, vel_aim
        B = self.get_B(fly_time)
        C = self.get_C(fly_time)
        return np.array([
            6 * ((tau-1)*(B-A) + (1-2*tau)*(C-B) + tau*(D-C)) / (fly_time**2)
            for tau in taus
        ])
    
    def get_fly_time_minimum(self):
        # TODO
        pass
    
    def get_fly_time_random(self):
        fly_time_A = ((self.A - self.D).len() / self.velA.len())
        fly_time_D = ((self.A - self.D).len() / self.velD.len())
        fly_time_min = min(fly_time_A, fly_time_D)
        fly_time_max = max(fly_time_A, fly_time_D)
        return np.random.uniform(fly_time_min, fly_time_max)
    
    def _fpos(self, t, fly_time=None):
        fly_time = fly_time if fly_time != None else self.fly_time
        A, velA = self.A, self.velA
        D, velD = self.D, self.velD
        B = self.get_B(fly_time)
        C = self.get_C(fly_time)
        tau = t / fly_time
        return (1-tau)**3 * A + 3*tau*(1-tau)**2 * B + 3*tau*tau*(1-tau)*C + tau**3 * D
    
    def _fvel(self, t, fly_time=None):
        fly_time = fly_time if fly_time != None else self.fly_time
        A, velA = self.A, self.velA
        D, velD = self.D, self.velD
        B = self.get_B(fly_time)
        C = self.get_C(fly_time)
        tau = t / fly_time
        return (3*(1-tau)**2*(B-A) + 6*tau*(1-tau)*(C-B) + 3*tau**2*(D-C)) / fly_time
    
    def _facc(self, t, fly_time=None):
        fly_time = fly_time if fly_time != None else self.fly_time
        A, velA = self.A, self.velA
        D, velD = self.D, self.velD
        B = self.get_B(fly_time)
        C = self.get_C(fly_time)
        tau = t / self.fly_time 
        return 6 * ((tau-1)*(B-A) + (1-2*tau)*(C-B) + tau*(D-C)) / (fly_time**2)
    
#     def get_amax(self, fly_time, vel_trg=None, g=Vec3(0, -9.80665, 0)):
#         vel_trg = self.velD if vel_trg == None else vel_trg
#         A, velA = self.pos, self.velA
#         D, velD = self.D, vel_trg
#         B = get_B(fly_time)
#         C = get_C(fly_time)
#         a1 = (C - B) * 6 - (B - A) * 6 - g
#         a2 = (D - C) * 6 - (C - B) * 6 - g
#         return np.fmax(a1.len(), a2.len()) / (fly_time**2)

#       cpdef (double, double) get_max_v_a(double delta_t, Vec2 A, Vec2 B, Vec2 C, Vec2 D, int n=33, int rounds=3, bint inc_g=False):
#     cdef Vec2 BA = B.sub_vec(A)
#     cdef Vec2 CB = C.sub_vec(B)
#     cdef Vec2 DC = D.sub_vec(C)
#     cdef double min_v = 1e13
#     cdef double max_v = 0
#     cdef int i, j, imax
#     cdef double dt = 1.0 / (n-1)
#     cdef double t, vel_len, t0, t1
#     t0 = 0.0
#     t1 = 1.0
#     for j in range(rounds):
#         dt = (t1-t0) / (n-1)
#         for i in range(n):
#             t = t0 + i * dt
#             vel_len = (BA.mul_num(3*(1-t)*(1-t)/delta_t).add_vec(CB.mul_num(6*t*(1-t)/delta_t)).add_vec(DC.mul_num(3*t*t/delta_t))).len() 
#             if vel_len > max_v:
#                 max_v = vel_len
#                 imax = i
#         t = t0 + imax * dt
#         t0 = t - dt
#         t1 = t + dt
#         if t0 < 0:
#             t0 = 0.0
#         if t1 > 1:
#             t1 = 1.0
#     cdef Vec2 g = Vec2(0, 0)
#     if inc_g:
#         g.y = -9.81
#     cdef Vec2 a1 = ((CB.mul_num(6)).sub_vec(BA.mul_num(6))).sub_vec(g)
#     cdef Vec2 a2 = ((DC.mul_num(6)).sub_vec(CB.mul_num(6))).sub_vec(g)
#     return max_v, fmax(a1.len(), a2.len())/delta_t/delta_t

#     cpdef (double, double) get_min_max_v(double delta_t, Vec2 A, Vec2 B, Vec2 C, Vec2 D, int n=42):
#     cdef Vec2 BA = B.sub_vec(A)
#     cdef Vec2 CB = C.sub_vec(B)
#     cdef Vec2 DC = D.sub_vec(C)
#     cdef double min_v = 1e13
#     cdef double max_v = 0
#     cdef int i
#     cdef double dt = 1.0 / (n-1)
#     cdef double t, vel_len
#     for i in range(n):
#         t = i * dt
#         vel_len = (BA.mul_num(3*(1-t)*(1-t)/delta_t).add_vec(CB.mul_num(6*t*(1-t)/delta_t)).add_vec(DC.mul_num(3*t*t/delta_t))).len() 
#         # print(t, vel_len)
#         if vel_len < min_v:
#             min_v = vel_len
#         if vel_len > max_v:
#             max_v = vel_len
#     return min_v, max_v
    
    def to_dict(self):
        return { 
            't': self.t,
            'x': self.x,
            'y': self.y,
            'z': self.z,
            'v': self.vel_abs,
            'vx': self.vel[0],
            'vy': self.vel[1],
            'vz': self.vel[2]
        }
    
    def _set_fly_time(self, time_min=True):
        self.fly_time = self.get_fly_time_minimum() if time_min else self.get_fly_time_random()
    
    @staticmethod
    def _validate_dimension_argument(array, n=3):
        if len(array) == 0:
            return Vec3(0,0,0)            
        elif len(array) == n:
            try:
                l = [float(elem) for elem in array]
            except ValueError:
                raise ValueError("Один или несколько элементов в <{!r}> не действительное(-ые) число(-а)".format(seq))
            else:
                arg1, arg2, arg3 = l
                return Vec3(arg1, arg2, arg3)  
        else:
            raise ValueError("Неожиданное число элементов. Получено: {}, Ожидалось: {}.".format(len(array), n))  