import numpy as np
import matplotlib.pyplot as plt
from casadi import *
import scipy.optimize as scopt

import time
import os

class Nelder_Mead:
    def __init__(self, OC_sys, reference_traj, reference_time, X0, step_size, num_iter):
        ''' Reference Trajectory '''
        self.y_r = reference_traj
        self.t_points = reference_time

        ''' OC System '''
        # A system defined by COC_sys
        self.sys = OC_sys
        # self.sys = Cont_COCSys()
        if not hasattr(self.sys, 'g'):
            self.sys.setMeasurement()

        # Details of parameters to be solved for
        self.m = self.sys.n_param

        ''' Algorithm Parameters '''
        # descent step-size
        self.alpha = step_size
        self.N = num_iter

        ''' Iteration Data Stuff '''
        self.Xk = np.zeros((self.N+1, X0.shape[0]))
        self.Xk[0] = X0

        # self.fXk = np.zeros(self.N+1)
        self.fXk = []

        self.X_sol = None
        self.x_sol_traj = None
        self.t_sol = None
        
        ''' Timers '''
        self.time_total = 0

        self.fig_states = None
        self.fig_error = None

    def loss_func(self, p):
        opt_sol = self.sys.ocSolver(param_value=p)
        loss = 0

        for k, t in enumerate(self.t_points):
            y = self.sys.g_func(opt_sol['x_traj_opt'](t), opt_sol['u_traj_opt'](t)).full().flatten()

            loss += np.linalg.norm(self.y_r[k]-y)**2

        return loss
    
    def callback(self, intermediate_result: scopt.OptimizeResult):
        self.fXk.append(intermediate_result.fun)

    def run(self):
        start_tot = time.perf_counter()

        self.fXk.append(self.loss_func(self.Xk[0]))

        opts = {'maxiter': self.N, 'return_all': True}
        res = scopt.minimize(self.loss_func, self.Xk[0], method='Nelder-Mead', bounds=scopt.Bounds(lb=0), options=opts, callback=self.callback)

        self.X_sol = res.x
        self.fXk.append(res.fun)

        sol = self.sys.ocSolver(param_value=self.X_sol)
        self.t_sol = sol['time']
        self.x_sol_traj = sol['x_traj_opt'](self.t_sol)

        end_tot = time.perf_counter()
        self.time_total = end_tot - start_tot

    ''' Saving Results '''
    def plot_results(self):
        # return if it hasn't been run
        # if self.Q_sol == None:
        #     return
        
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        
        if self.sys.n_state!=2 and self.sys.n_state//2 == self.sys.n_state/2:
            self.fig_states, ax_states = plt.subplots(self.sys.n_state//2,2,layout='constrained',sharex='col', figsize=[6.4,4])
            for i in range(self.sys.n_state):
                # q = i//(n//2)
                # p = (i-q*(n//2))
                p = i//2
                q = i-p-i//(2)

                if i == 0:
                    ax_states[p][q].plot(self.t_sol, self.x_sol_traj.T[i], "-", color="tab:blue", linewidth=2, label=r"$x_{\hat{X}}$")
                    # ax_states[p][q].plot(self.t_sol, self.y_r.T[i], '--', color="tab:red", linewidth=1, label=r"$x^*$")
                    
                    ax_states[p][q].legend()
                else:
                    ax_states[p][q].plot(self.t_sol, self.x_sol_traj.T[i], "-", color="tab:blue", linewidth=2)
                    # ax_states[p][q].plot(self.t_sol, self.y_r.T[i], '--', color="tab:red", linewidth=1)

                ax_states[p][q].set_ylabel(fr"$x_{i+1}(t)$")
            
            ax_states[-1][0].set_xlabel(r"$t$")
            ax_states[-1][1].set_xlabel(r"$t$")

        else:
            if self.sys.n_state == 2:
                self.fig_states, ax_states = plt.subplots(1, self.sys.n_state, layout='constrained', sharex='col')
            else:
                self.fig_states, ax_states = plt.subplots(self.sys.n_state,layout='constrained',sharex='col')
                ax_states[-1].set_xlabel(r"$t$")

            for i in range(self.sys.n_state):
                if i == 0:
                    ax_states[i].plot(self.t_sol, self.x_sol_traj.T[i], "-", color="tab:blue", linewidth=2, label=r"$x_{\hat{X}}$")
                    # ax_states[i].plot(self.t_sol, self.y_r.T[i], '--', color="tab:red", linewidth=1, label=r"$x^*$")
                    
                    ax_states[i].legend()
                else:
                    ax_states[i].plot(self.t_sol, self.x_sol_traj.T[i], "-", color="tab:blue", linewidth=2)
                    # ax_states[i].plot(self.t_sol, self.y_r.T[i], '--', color="tab:red", linewidth=1)
                
                if self.sys.n_state == 2:
                    ax_states[i].set_xlabel(r"$t$")

                ax_states[i].set_ylabel(fr"$x_{i+1}(t)$")

        # self.fig_states.suptitle("State Trajectories")

        # plot error function
        self.fig_error, ax_error = plt.subplots(layout='constrained', figsize=[4.34,4])
        ax_error.plot(np.arange(0,self.fXk.shape[0]), self.fXk)
        ax_error.set_ylabel(r"$f(X_k)$")
        ax_error.set_xlabel(r"$k$")
        # ax_error.set_title("Error Function per Iteration")

        ax_error.text(0.01, 0.05, f'{(self.fXk[0]-np.min(self.fXk)):0.2f} Absolute', horizontalalignment='left', verticalalignment='bottom', transform=ax_error.transAxes)
        ax_error.text(0.01, 0.05, f'{100*(self.fXk[0]-np.min(self.fXk))/self.fXk[0]:0.2f}\%  Improvement', horizontalalignment='left', verticalalignment='top', transform=ax_error.transAxes)

        # plt.show()
    
    def save_results(self, env, exp=None, true_X=None, plots=False, txt=False, npz=False):
        # create a folder to save the results
        folder_name = "results/naive/"+env
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        if plots:
            folder_name2 = "results/naive/"+env+str(exp or '')
            if not os.path.exists(folder_name2):
                os.makedirs(folder_name2)
            # save figures
            self.fig_states.savefig(folder_name2+"/state.pdf",bbox_inches='tight')
            self.fig_error.savefig(folder_name2+"/error.pdf",bbox_inches='tight')

        if txt:
            folder_name2 = "results/naive/"+env+str(exp or '')
            if not os.path.exists(folder_name2):
                os.makedirs(folder_name2)
            # save values
            # set np precision
            np.set_printoptions(precision=3)

            filename = folder_name2+"/params.txt"
            file = open(filename, "w+")

            # reference config
            file.write("--- Reference States --- \n")
            if self.include_u:
                file.write("x0 = " + str(self.x0[0:self.sys.n_state]) + "\n")
            else:
                file.write("x0 = " + str(self.x0) + "\n")
            file.write(f"No. timesteps = {self.y_r.shape[0]} \n")
            file.write(f"T = {self.t_sol[-1]:0.2f} \n")
            file.write("\n")


            # algorithm parameters
            file.write("--- parameters --- \n")
            file.write(f"N = {self.N}, ")
            file.write(f"alpha = {self.alpha[0]:0.2e} \n")
            file.write("X0 = diag("+str(self.Xk[0])+") \n")
            file.write("\n")

            # results
            file.write("--- Results --- \n")
            file.write(f"f_X0 = {self.fXk[0]:0.4f}, f_min = {np.min(self.fXk):0.4f} \n")
            file.write(f"Percentage Improvement = {100*(self.fXk[0]-np.min(self.fXk))/self.fXk[0]:0.2f}% \n")
            file.write(f"Absolute Reduction = {self.fXk[0]-np.min(self.fXk):0.2f} \n")
            file.write("X* = diag("+str(self.X_sol)+") \n")
            file.write(" \n")

            file.write("--- Timing --- \n")
            if self.time_total>120:
                file.write(f"Total time = {self.time_total/60:0.2f} mins \n")
            else:
                file.write(f"Total time = {self.time_total:0.2f}s \n")


            file.write(" \n")
            file.write(" \n")
            file.write("fXk = \n")
            file.write(str(self.fXk))

            file.close()

        if npz:
            np.savez(folder_name+str(exp or '')+"_params", 
                        error=self.fXk, 
                        X0=self.Xk[0], 
                        X_sol=self.X_sol, 
                        X_true=true_X, 
                        alpha=self.alpha, 
                        time=self.time_total,
                        num_iter=self.N)
