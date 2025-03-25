from casadi import *
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
import os
import time

class CPDP_IOC:
    def __init__(self, OC_sys, reference_traj, reference_time, x0, horizon, init_params, learning_rate, num_iter, measurement_func=None):
        ''' Reference Trajectory '''
        self.yr = reference_traj
        self.horizon = horizon
        self.t_points = reference_time

        ''' OC System '''
        # A system defined by PDP.OCSys()
        self.sys = OC_sys
        self.sys.diffPMP()
        # Details of parameters to be solved for
        self.m = self.sys.n_auxvar
        self.x0 = x0

        if measurement_func is None:
            if self.yr.shape[1] == OC_sys.n_state + OC_sys.n_control:
                self.g = vertcat(self.sys.state, self.sys.control)
            else:
                self.g = self.sys.state
        else:
            self.g = measurement_func
        
        self.g_func = Function('g', [self.sys.state, self.sys.control], [self.g])
        self.dg_dx_func = Function('dg_dx', [self.sys.state, self.sys.control], [jacobian(self.g, vertcat(self.sys.state, self.sys.control))])

        ''' Algorithm Stuff '''
        self.lr = learning_rate
        self.N = num_iter

        ''' Iteration Data '''
        # self.X0 = init_params
        self.Xk = np.zeros((self.N+1, init_params.shape[0]))
        self.Xk[0] = init_params

        self.loss = np.zeros(self.N+1)

        self.X_sol = None
        self.x_sol_traj = None
        self.t_sol = None

        ''' Other stuff to save '''
        self.time_total = 0
        self.fig_states = None
        self.fig_error = None

    def run(self):

        start_tot = time.perf_counter()


        for k in tqdm(range(self.N)):
            # learn current traj from current parameter guess
            opt_sol = self.sys.cocSolver(self.x0, self.horizon, self.Xk[k])

            # establish auxiliary sys
            aux_sol = self.sys.auxSysSolver(opt_sol, self.Xk[k])

            self.loss[k], dp = self.evaluate_loss(opt_sol, aux_sol)
            
            # parameter update
            self.Xk[k+1] = self.Xk[k] - self.lr[k]*dp
        
        # compute loss for final iteration
        opt_sol = self.sys.cocSolver(self.x0, self.horizon, self.Xk[self.N])
        aux_sol = self.sys.auxSysSolver(opt_sol, self.Xk[self.N])
        self.loss[self.N], _ = self.evaluate_loss(opt_sol, aux_sol)

        # Find optimal solution from lowest error function value
        k = np.argmin(self.loss)
        self.X_sol = self.Xk[k]

        # Compute the state trajectory using optimal solution
        sol = self.sys.cocSolver(ini_state=self.x0, horizon=self.horizon, auxvar_value=self.X_sol)
        self.t_sol = sol['time']
        self.x_sol_traj = sol['state_traj_opt'](self.t_sol)

        end_tot = time.perf_counter()
        self.time_total = end_tot - start_tot

    def evaluate_loss(self, opt_sol, aux_sol):
        loss = 0
        dldp = np.zeros(self.sys.n_auxvar)

        for k, t in enumerate(self.t_points):
            y = self.g_func(opt_sol['state_traj_opt'](t), opt_sol['control_traj_opt'](t)).full().flatten()

            loss += np.linalg.norm(self.yr[k]-y)**2

            dl_dg = -2*(self.yr[k]-y) # 1 x q
            dg_dx = self.dg_dx_func(opt_sol['state_traj_opt'](t), opt_sol['control_traj_opt'](t)).full() # q x (n+m)
            dx_dp = aux_sol(t).reshape((self.sys.n_state+self.sys.n_control, self.sys.n_auxvar)) # (n+m) x p

            dldp += np.matmul(dl_dg, np.matmul(dg_dx, dx_dp))

        return loss, dldp


    def plot_results(self):        
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
                    # ax_states[p][q].plot(self.t_sol, self.yr.T[i], '--', color="tab:red", linewidth=1, label=r"$x^*$")
                    
                    ax_states[p][q].legend()
                else:
                    ax_states[p][q].plot(self.t_sol, self.x_sol_traj.T[i], "-", color="tab:blue", linewidth=2)
                    # ax_states[p][q].plot(self.t_sol, self.yr.T[i], '--', color="tab:red", linewidth=1)

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
                    # ax_states[i].plot(self.t_sol, self.yr.T[i], '--', color="tab:red", linewidth=1, label=r"$x^*$")
                    
                    ax_states[i].legend()
                else:
                    ax_states[i].plot(self.t_sol, self.x_sol_traj.T[i], "-", color="tab:blue", linewidth=2)
                    # ax_states[i].plot(self.t_sol, self.yr.T[i], '--', color="tab:red", linewidth=1)
                
                if self.sys.n_state == 2:
                    ax_states[i].set_xlabel(r"$t$")

                ax_states[i].set_ylabel(fr"$x_{i+1}(t)$")

        # self.fig_states.suptitle("State Trajectories")

        # plot error function
        self.fig_error, ax_error = plt.subplots(layout='constrained', figsize=[4.34,4])
        ax_error.plot(np.arange(0,self.loss.shape[0]), self.loss)
        ax_error.set_ylabel(r"$f(X_k)$")
        ax_error.set_xlabel(r"$k$")
        # ax_error.set_title("Error Function per Iteration")

        ax_error.text(0.01, 0.05, f'{(self.loss[0]-np.min(self.loss)):0.2f} Absolute', horizontalalignment='left', verticalalignment='bottom', transform=ax_error.transAxes)
        ax_error.text(0.01, 0.05, f'{100*(self.loss[0]-np.min(self.loss))/self.loss[0]:0.2f}\%  Improvement', horizontalalignment='left', verticalalignment='top', transform=ax_error.transAxes)
    
    def save_results(self, env, exp=None, true_X=None, plots=False, txt=False, npz=False):
        # create a folder to save the results
        folder_name = "results/pdp/"+env
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        
        if plots:
            folder_name2 = "results/pdp/"+env+str(exp or '')
            if not os.path.exists(folder_name2):
                os.makedirs(folder_name2)
            # save figures
            self.fig_states.savefig(folder_name2+"/state.pdf",bbox_inches='tight')
            self.fig_error.savefig(folder_name2+"/error.pdf",bbox_inches='tight')

        if txt:
            folder_name2 = "results/pdp/"+env+str(exp or '')
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
            file.write(f"No. timesteps = {self.yr.shape[0]} \n")
            file.write(f"T = {self.t_sol[-1]:0.2f} \n")
            file.write("\n")


            # algorithm parameters
            file.write("--- Algorithm parameters --- \n")
            file.write(f"N = {self.N}, ")
            file.write(f"learning rate = {self.lr[0]:0.2e} \n ")
            file.write("X0 = "+str(self.Xk[0])+" \n")
            file.write("\n")

            # results
            file.write("--- Results --- \n")
            file.write(f"f_X0 = {self.loss[0]:0.4f}, f_min = {np.min(self.loss):0.4f} \n")
            file.write(f"Percentage Improvement = {100*(self.loss[0]-np.min(self.loss))/self.loss[0]:0.2f}% \n")
            file.write(f"Absolute Reduction = {self.loss[0]-np.min(self.loss):0.2f} \n")
            file.write("X* = "+str(self.X_sol)+" \n")
            file.write(" \n")

            file.write("--- Timing --- \n")
            if self.time_total>120:
                file.write(f"Total time = {self.time_total/60:0.2f} mins \n")
            else:
                file.write(f"Total time = {self.time_total:0.2f}s \n")
            
            file.write(" \n")
            file.write(" \n")
            file.write("loss = \n")
            file.write(str(self.loss))

            file.close()

        if npz:
            # save stuff as numpy arrays
            np.savez(folder_name+str(exp or '')+"_params", 
                     error=self.loss,
                     X0=self.Xk[0],
                     Xk=self.Xk, 
                     X_sol=self.X_sol,
                     X_true=true_X, 
                     learning_rate=self.lr,
                     time=self.time_total)
    