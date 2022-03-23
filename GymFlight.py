import numpy as np
from Missile3D import Missile3D
from Target3D import Target3D
import matplotlib.pyplot as plt


class GymFlight(object):

    @classmethod
    def make_simple_scenario(cls,
                             missile_opts,
                             target_opts,
                             tau=0.5,
                             ndt=10,
                             n_step=50,
                             t_max=50,
                             fly_time_min=False,
                             postProcessing=True):
        mis = Missile3D.get_missile(missile_opts,
                                    postProcessing=postProcessing)
        trg = Target3D.get_simple_target(target_opts,
                                         time_min=fly_time_min,
                                         postProcessing=postProcessing)
        mis.set_initial_parameters_of_missile(trg)
        return cls(mis=mis, trg=trg, n_step=n_step, tau=tau, ndt=ndt, t_max=t_max)

    def __init__(self, **kwargs):
        self.missile = kwargs['mis']
        self.target = kwargs['trg']
        self.i_step = 0
        self.n_step = kwargs['n_step']
        self.tau = kwargs['tau']
        self.ndt = kwargs['ndt']
        self.t_max = kwargs['t_max']
        self.history = None

    def run(self):
        done = False
        while not done:
            self.step()
            self.get_info_about_step()
            self.i_step += 1
            done, info = self.get_info_about_step()
        print(f'Info launch missile: {info}')
        self.history = {'missile': self.missile.history, 'target': self.target.history} if self.missile.postProcessing else self.get_state()

    def reset(self):
        self.missile.reset()
        self.target.reset()

    def step(self):
        self.target.step(tau=self.tau, n=self.ndt)
        self.missile.step(self.missile.get_action_proportional_guidance(self.target), tau=self.tau, n=self.ndt)

    def get_info_about_step(self):
        if self._r_() < self.missile.r_explosion:
            return True, 'target defeat'
        elif self.missile.t > self.t_max:
            return True, 'long fly time'
        elif self.i_step > self.n_step:
            return True, 'count step max'
        else:
            return False, ''

    def get_state(self):
        return {'missile': self.missile.state,
                'target': self.target.state}

    def to_dict(self):
        return {'missile': self.missile.to_dict(),
                'target': self.target.to_dict()}

    def plot(self, dpi=150,
             ls_trg_full='--', ls_trg='-', ls_mis='-',
             color_trg_full='darkorange', color_trg='red', color_mis='darkblue',
             marker_meet='o', color_marker_meet='red', marker_size=15,
             marker_trg='x', color_marker_trg='red', color_marker_size=30,
             loc_legend='upper left', legend_size=8,
             savefig=False,
             ):
        traj = self.target.get_traject(self.target.fly_time)
        res = self.to_dict()

        plt.figure(dpi=dpi)
        ax = plt.axes(projection='3d')

        ax.plot(traj[:, 0], traj[:, 2], traj[:, 1],
                ls=ls_trg_full, color=color_trg_full, label='Полная траектория цели')
        ax.plot(res['target']['x'], res['target']['z'], res['target']['y'],
                ls=ls_trg, color=color_trg, label='Траектория цели')
        ax.plot(res['missile']['pos'][:, 0], res['missile']['pos'][:, 2], res['missile']['pos'][:, 1],
                ls=ls_mis, color=color_mis, label='Траектория ракеты')
        ax.scatter(res['target']['x'][-1], res['target']['z'][-1], res['target']['y'][-1],
                   marker=marker_meet, color=color_marker_meet, s=marker_size, label='Точка встречи')
        ax.scatter(traj[:, 0][-1], traj[:, 2][-1], traj[:, 1][-1],
                   marker=marker_trg, color=color_marker_trg, s=color_marker_size, label='Конечное положение цели')

        ax.set_xlabel('$X$, м')
        ax.set_ylabel('$Z$, м')
        ax.set_zlabel('$Y$, м')
        ax.view_init(elev=20, azim=-150)
        plt.legend(loc=loc_legend, fontsize=legend_size)

        if savefig:
            plt.savefig('scenario_render.jpg')

        plt.show()

    def plot_trajectory(self, figsize=(15, 5), fontsize=14, labelsize=12, savefig=False):
        res = self.to_dict()

        fig = plt.figure(figsize=figsize)
        ax_1 = fig.add_subplot(1, 3, 1)
        ax_2 = fig.add_subplot(1, 3, 2)
        ax_3 = fig.add_subplot(1, 3, 3)

        ax_1.plot(res['missile']['pos'][:, 0], res['missile']['pos'][:, 1], label='Траектория ракеты')
        ax_1.plot(res['target']['x'], res['target']['y'], label='Траектория цели')

        ax_2.plot(res['missile']['pos'][:, 2], res['missile']['pos'][:, 0])
        ax_2.plot(res['target']['z'], res['target']['x'])

        ax_3.plot(res['missile']['pos'][:, 2], res['missile']['pos'][:, 1])
        ax_3.plot(res['target']['z'], res['target']['y'])

        ax_1.set_xlabel('$x$, м', fontsize=fontsize)
        ax_1.set_ylabel('$y$, м', fontsize=fontsize)

        ax_2.set_xlabel('$z$, м', fontsize=fontsize)
        ax_2.set_ylabel('$x$, м', fontsize=fontsize)

        ax_3.set_xlabel('$z$, м', fontsize=fontsize)
        ax_3.set_ylabel('$y$, м', fontsize=fontsize)

        ax_1.set(title='X0Y')
        ax_2.set(title='Z0X')
        ax_3.set(title='Z0Y')

        ax_1.legend(fontsize=fontsize)
        ax_1.tick_params(labelsize=labelsize)
        ax_2.tick_params(labelsize=labelsize)
        ax_3.tick_params(labelsize=labelsize)

        if savefig:
            plt.savefig('scenario_projection.jpg')

        plt.show()

    def plot_motion_parameters(self, figsize=(15, 7), fontsize=14, labelsize=12, dpi=400, savefig=False):
        fig = plt.figure(figsize=figsize)

        ax_1 = fig.add_subplot(2, 2, 1)
        ax_2 = fig.add_subplot(2, 2, 2)
        ax_3 = fig.add_subplot(2, 2, 3)
        ax_4 = fig.add_subplot(2, 2, 4)

        ax_1.plot(self.missile.history['t'], self.missile.history['v_abs'], label='$V(t)$')

        ax_2.plot(self.missile.history['t'], np.degrees(self.missile.history['thetta']), label='θ')
        ax_2.plot(self.missile.history['t'], np.degrees(self.missile.history['psi']), label='ψ')
        ax_2.plot(self.missile.history['t'], np.degrees(self.missile.history['gamma']), label='γ')

        ax_3.plot(self.missile.history['t'], self.missile.history['qw'], label='$q_w$')
        ax_3.plot(self.missile.history['t'], self.missile.history['qx'], label='$q_x$')
        ax_3.plot(self.missile.history['t'], self.missile.history['qy'], label='$q_y$')
        ax_3.plot(self.missile.history['t'], self.missile.history['qz'], label='$q_z$')

        ax_4.plot(self.missile.history['t'], self.missile.history['alpha'], label='α')
        ax_4.plot(self.missile.history['t'], self.missile.history['betta'], label='β')
        ax_4.plot(self.missile.history['t'], self.missile.history['alpha_targeting'], label='α$_{targeting}$')
        ax_4.plot(self.missile.history['t'], self.missile.history['betta_targeting'], label='β$_{targeting}$')

        ax_1.set_ylabel('$V$, м/c', fontsize=fontsize)
        ax_2.set_ylabel('$angle$, град', fontsize=fontsize)
        ax_3.set_xlabel('$t$, c', fontsize=fontsize)
        ax_3.set_ylabel('$q$', fontsize=fontsize)
        ax_4.set_xlabel('$t$, c', fontsize=fontsize)
        ax_4.set_ylabel('$angle$, град', fontsize=fontsize)

        ax_1.set_title(label='Профиль скорости', fontdict={'fontsize': fontsize})
        ax_2.set_title(label='Углы ориентации ракеты', fontdict={'fontsize': fontsize})
        ax_3.set_title(label='Кватернионы', fontdict={'fontsize': fontsize})
        ax_4.set_title(label='Управляющие углы', fontdict={'fontsize': fontsize})

        ax_1.legend(fontsize=fontsize)
        ax_2.legend(fontsize=fontsize)
        ax_3.legend(fontsize=fontsize, loc='center left')
        ax_4.legend(fontsize=fontsize)

        ax_1.tick_params(labelsize=labelsize)
        ax_2.tick_params(labelsize=labelsize)
        ax_3.tick_params(labelsize=labelsize)
        ax_4.tick_params(labelsize=labelsize)

        if savefig:
            plt.savefig('scenario_motion.jpg', dpi=dpi)

        plt.show()

    def plot_forces_parameters(self, figsize=(15, 7), fontsize=14, labelsize=12, dpi=400, savefig=False):
        fig = plt.figure(figsize=figsize)

        ax_1 = fig.add_subplot(1, 3, 1)
        ax_2 = fig.add_subplot(1, 3, 2)
        ax_3 = fig.add_subplot(1, 3, 3)

        ax_1.plot(self.missile.history['t'], self.missile.history['X'], label='$X$')
        ax_1.plot(self.missile.history['t'], self.missile.history['Y'], label='$Y$')
        ax_1.plot(self.missile.history['t'], self.missile.history['Z'], label='$Z$')

        ax_2.plot(self.missile.history['t'], self.missile.history['Mx'], label='$M_x$')
        ax_2.plot(self.missile.history['t'], self.missile.history['My'], label='$M_y$')
        ax_2.plot(self.missile.history['t'], self.missile.history['Mz'], label='$M_z$')

        ax_3.plot(self.missile.history['t'], self.missile.history['wx'], label='ω$_x$')
        ax_3.plot(self.missile.history['t'], self.missile.history['wy'], label='ω$_y$')
        ax_3.plot(self.missile.history['t'], self.missile.history['wz'], label='ω$_z$')

        ax_1.set_ylabel('$F$, Н', fontsize=fontsize)
        ax_2.set_ylabel('$M$, Н∙м', fontsize=fontsize)
        ax_3.set_ylabel('ω, рад/с', fontsize=fontsize)
        ax_1.set_xlabel('$t$, c', fontsize=fontsize)
        ax_2.set_xlabel('$t$, c', fontsize=fontsize)
        ax_3.set_xlabel('$t$, c', fontsize=fontsize)

        ax_1.set_title(label='Силы в ССК', fontdict={'fontsize': fontsize})
        ax_2.set_title(label='Моменты в ССК', fontdict={'fontsize': fontsize})
        ax_3.set_title(label='Угловые скорости в ССК', fontdict={'fontsize': fontsize})

        ax_1.legend(fontsize=fontsize)
        ax_2.legend(fontsize=fontsize)
        ax_3.legend(fontsize=fontsize)

        ax_1.tick_params(labelsize=labelsize)
        ax_2.tick_params(labelsize=labelsize)
        ax_3.tick_params(labelsize=labelsize)

        if savefig:
            plt.savefig('scenario_forces.jpg', dpi=dpi)

        plt.show()

    def _r_(self):
        return np.sqrt((self.target.pos[0] - self.missile.pos[0]) ** 2 + \
                       (self.target.pos[1] - self.missile.pos[1]) ** 2 + \
                       (self.target.pos[2] - self.missile.pos[2]) ** 2)