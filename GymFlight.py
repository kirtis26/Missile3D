import numpy as np
from easyvec import Vec3
from Missile3D import Missile3D
from Target3D import Target3D


class GymFlight(object):

    @classmethod
    def make_simple_scenario(cls,
                             missile_opts,
                             target_opts,
                             n_step=30,
                             tau=0.5,
                             ndt=10,
                             fly_time_min=False,
                             postProcessing=True):
        mis = Missile3D.get_missile(missile_opts,
                                    postProcessing=postProcessing)
        trg = Target3D.get_simple_target(target_opts,
                                         time_min=fly_time_min,
                                         postProcessing=postProcessing)
        mis.set_initial_parameters_of_missile(trg)
        return cls(mis=mis, trg=trg, n_step=n_step, tau=tau, ndt=ndt)

    def __init__(self, **kwargs):
        self.missile = kwargs['mis']
        self.target = kwargs['trg']
        self.i_step = 0
        self.n_step = kwargs['n_step']
        self.tau = kwargs['tau']
        self.ndt = kwargs['ndt']

    def run(self):
        while self._r_() < self.missile.r_explosion or self.i_step < self.n_step:
            self.step()
            self.get_info_about_step()
            self.i_step += 1

    def reset(self):
        self.missile.reset()
        self.target.reset()

    def step(self):
        self.target.step(tau=self.tau, n=self.ndt)
        self.missile.step(self.missile.get_action_proportional_guidance(self.target), tau=self.tau, n=self.ndt)

    def get_info_about_step(self):
        # TODO
        pass

    def get_state(self):
        # TODO
        pass

    def _r_(self):
        return np.sqrt((self.target.pos[0] - self.missile.pos[0]) ** 2 + \
                       (self.target.pos[1] - self.missile.pos[1]) ** 2 + \
                       (self.target.pos[2] - self.missile.pos[2]) ** 2)


trg_opts = {
    'pos': Vec3(8e3, 1.5e3, 0),
    'pos_aim': Vec3(-100, 0, 100),
    'vel': Vec3(-100, 10, 10),
    'vel_aim': Vec3(-100, -10, -10)
}

mis_opts = {
    'd': 0.230,
    'L': 2.89,
    'm_0': 165,
    'v_0': 25,
    't_marsh': 6,
    'w_marsh': 53,
    'P_marsh': 22.5 * 1e3,
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
