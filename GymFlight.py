from Missile3D import Missile3D
from Target3D import Target3D
import matplotlib.pyplot as plt
import numpy as np

# TODO: класс запуска ракеты 

opts = {
    'd': 0.230,
    'L': 2.89,
    'm_0': 165,
    'v_0': 25,
    't_marsh': 6,
    'w_marsh': 53,
    'P_marsh': 22.5*1e3,
    'r_explosion': 50,
    'angle_max': 12,
    'a_m': 3,
    'record': True,
    'x_cm_0': 1.81,
    'x_cm_k': 1.7,
    'x_pres_0': 1.87,
    'x_pres_k': 1.8,
    'x_rule': 0.6,
    'alpha': np.array([-10, -7, -5, -3, -1, 0, 1, 3, 5, 7, 10]),
    'Mach': np.array([0.3, 0.6, 0.9, 1, 1.1, 1.4, 1.7, 2.0, 2.3]),
    'delta': np.array([-15, 0, 15]),
    'Cy_alpha': np.array([0.2, 0.3, 0.35, 0.4, 0.35, 0.3, 0.25, 0.225, 0.2]),
    'Cy_delta': np.array([0.10, 0.15, 0.20, 0.25, 0.20, 0.15, 0.12, 0.11, 0.10]),
    'mz_wz': np.array([-30, -35, -40, -40, -35, -30, -25, -20, -15]),
    'mx_wx': np.array([-0.10, -0.15, -0.20, -0.25, -0.30, -0.35, -0.40, -0.45, -0.50]),
    'mx_delta': np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]),
    'i': 1.0
}

mis = Missile3D.get_missile(opts)

trg = Target3D.get_simple_target(pos=Vec3(8e3, 1.5e3, 0),
                                 aim=Vec3(-100, 0, 100),
                                 vel=Vec3(-100, 10, 10),
                                 vel_aim=Vec3(-100, -10, -10),
                                 time_min=False)
traj = trg.get_traject(trg.fly_time)
mis.set_initial_parameters_of_missile(trg)

dt = 0.5
n = 10
for i in tqdm(range(30)):
    trg.step(tau=dt, n=n)
    mis.step(mis.get_action_proportional_guidance(trg), tau=dt, n=n)
    if np.sqrt((trg.pos[0]-mis.pos[0])**2 + (trg.pos[1]-mis.pos[1])**2 + (trg.pos[2]-mis.pos[2])**2) < mis.r_explosion:
        break