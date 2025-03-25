from PDP import PDP

from casadi import *
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
import os
import time

class PDP_IOC:
    def __init__(self, OC_sys, reference_traj, init_params, learning_rate, num_iter):
        ''' Reference Trajectory '''
        self.xr = reference_traj
        self.horizon = reference_traj.shape[0] - 1

        # check if comparing u traj too
        if self.xr.shape[1] == OC_sys.n_state + OC_sys.n_control:
            self.include_u = True
            self.x0 = reference_traj[0, 0:OC_sys.n_state]
        elif self.xr.shape[1] == OC_sys.n_state:
            self.include_u = False
            self.x0 = reference_traj[0]

        ''' OC System '''
        # A system defined by PDP.OCSys()
        self.sys = OC_sys
        self.sys.diffPMP()
        # Details of parameters to be solved for
        self.m = self.sys.n_auxvar

        ''' Algorithm Stuff '''
        self.lr = learning_rate
        self.N = num_iter
        self.lqr_solver = PDP.LQR()

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
            traj = self.sys.ocSolver(self.x0, self.horizon, self.Xk[k])

            # establish auxiliary sys
            aux_sys = self.sys.getAuxSys(
                state_traj_opt=traj['state_traj_opt'],
                control_traj_opt=traj['control_traj_opt'],
                costate_traj_opt=traj['costate_traj_opt'],
                auxvar_value=self.Xk[k]
            )

            self.lqr_solver.setDyn(dynF=aux_sys['dynF'], dynG=aux_sys['dynG'], dynE=aux_sys['dynE'])
            self.lqr_solver.setPathCost(Hxx=aux_sys['Hxx'], Huu=aux_sys['Huu'], Hxu=aux_sys['Hxu'], Hux=aux_sys['Hux'], Hxe=aux_sys['Hxe'], Hue=aux_sys['Hue'])
            self.lqr_solver.setFinalCost(hxx=aux_sys['hxx'], hxe=aux_sys['hxe'])
            aux_sol = self.lqr_solver.lqrSolver(numpy.zeros((self.sys.n_state, self.sys.n_auxvar)), self.horizon)

            # take solution of the auxiliary control system
            dxdp_traj = aux_sol['state_traj_opt']
            dudp_traj = aux_sol['control_traj_opt']

            
            dp = np.zeros(self.Xk[k].shape)
            state_traj = traj['state_traj_opt']
            control_traj = traj['control_traj_opt']
            dldx_traj = 2*(state_traj - self.xr[:,:self.sys.n_state])
            if self.include_u:
                dldu_traj = 2*(control_traj - self.xr[:self.horizon,self.sys.n_state:])
            for t in range(self.horizon):
                if self.include_u:
                    dp = dp + np.matmul(dldx_traj[t, :], dxdp_traj[t]) + np.matmul(dldu_traj[t, :], dudp_traj[t])
                else:
                    dp = dp + np.matmul(dldx_traj[t, :], dxdp_traj[t]) 
            dp = dp + numpy.dot(dldx_traj[-1, :], dxdp_traj[-1])

            # evaluate loss
            if self.include_u:
                x_traj = np.concatenate((traj['state_traj_opt'], np.append(traj['control_traj_opt'],np.zeros((1,traj['control_traj_opt'].shape[1])), axis=0)), axis=1)
            else:
                x_traj = traj['state_traj_opt']
            self.loss[k] = np.linalg.norm(x_traj - self.xr)**2

            # parameter update
            self.Xk[k+1] = self.Xk[k] - self.lr[k]*dp
        
        # compute loss for final iteration
        traj = self.sys.ocSolver(self.x0, self.horizon, self.Xk[self.N])
        if self.include_u:
            x_traj = np.concatenate((traj['state_traj_opt'], np.append(traj['control_traj_opt'],np.zeros((1,traj['control_traj_opt'].shape[1])), axis=0)), axis=1)
        else:
            x_traj = traj['state_traj_opt']
        self.loss[self.N] = np.linalg.norm(x_traj - self.xr)**2

        # Find optimal solution from lowest error function value
        k = np.argmin(self.loss)
        self.X_sol = self.Xk[k]

        # Compute the state trajectory using optimal solution
        sol = self.sys.ocSolver(ini_state=self.x0, horizon=self.horizon, auxvar_value=self.X_sol)
        self.x_sol_traj = sol['state_traj_opt']
        self.t_sol = sol['time']

        end_tot = time.perf_counter()
        self.time_total = end_tot - start_tot

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
                    ax_states[p][q].plot(self.t_sol, self.xr.T[i], '--', color="tab:red", linewidth=1, label=r"$x^*$")
                    
                    ax_states[p][q].legend()
                else:
                    ax_states[p][q].plot(self.t_sol, self.x_sol_traj.T[i], "-", color="tab:blue", linewidth=2)
                    ax_states[p][q].plot(self.t_sol, self.xr.T[i], '--', color="tab:red", linewidth=1)

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
                    ax_states[i].plot(self.t_sol, self.xr.T[i], '--', color="tab:red", linewidth=1, label=r"$x^*$")
                    
                    ax_states[i].legend()
                else:
                    ax_states[i].plot(self.t_sol, self.x_sol_traj.T[i], "-", color="tab:blue", linewidth=2)
                    ax_states[i].plot(self.t_sol, self.xr.T[i], '--', color="tab:red", linewidth=1)
                
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
            file.write(f"No. timesteps = {self.xr.shape[0]} \n")
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
                     X_sol=self.X_sol,
                     X_true=true_X, 
                     learning_rate=self.lr,
                     time=self.time_total)
    