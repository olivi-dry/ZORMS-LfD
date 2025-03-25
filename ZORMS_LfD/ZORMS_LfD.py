from ZORMS_LfD.utils import project_cone

import numpy as np
import matplotlib.pyplot as plt
from casadi import *

import time
import os
from tqdm import tqdm

class ZORMS_disc:
    def __init__(self, OC_sys, reference_traj, X0, step_size, oracle_prec, num_iter, grad_protection_threshold = 1e5, lipschitz_check=False, RMSE=False):
        
        ''' Reference Trajectory '''
        self.x_r = reference_traj
        self.horizon = reference_traj.shape[0] - 1

        # check if data includes input traj
        if self.x_r.shape[1] == OC_sys.n_state + OC_sys.n_control:
            self.include_u = True
            self.x0 = reference_traj[0, 0:OC_sys.n_state]
        elif self.x_r.shape[1] == OC_sys.n_state:
            self.include_u = False
            self.x0 = reference_traj[0]

        ''' OC System '''
        # A system defined by PDP.OCSys() or SafePDP.COCSys()
        self.sys = OC_sys
        # Details of parameters to be solved for
        self.m = self.sys.n_auxvar

        ''' Algorithm Parameters '''
        self.alpha = step_size
        self.mu = oracle_prec
        self.N = num_iter

        self.RMSE = RMSE

        # ignore steps where grad estimate is too large
        self.grad_thresh = grad_protection_threshold 

        ''' Iteration Data '''
        self.Xk = np.zeros((self.N+1, X0.shape[0], X0.shape[0]))
        self.Xk[0] = X0

        self.fXk = np.zeros(self.N+1)

        self.X_sol = None
        self.x_sol_traj = None
        self.t_sol = None
        
        ''' Timers '''
        self.time_total = 0
        self.time_GOE = 0
        self.time_sys = 0
        self.time_proj = 0

        self.fig_states = None
        self.fig_error = None

        # For Lipschitz test
        self.lipschitz_check = lipschitz_check
        if self.lipschitz_check:
            self.L0 = None
            self.norm = np.zeros(self.N+1)
            self.fX_plus = np.zeros(self.N+1)
            self.r = 0
    
    def run(self):
        start_tot = time.perf_counter()

        prev_grad = 0
            
        for k in tqdm(range(self.N)):
            start = time.perf_counter()
            # Generate Random Direction
            U = np.diag(np.random.standard_normal(self.m))
            end = time.perf_counter()
            self.time_GOE += end-start

            # Compute Oracle
            if self.lipschitz_check:
                O, fX, fX_plus, norm = self.oracle(self.Xk[k], U)
                self.fX_plus[k] = fX_plus
                self.norm[k] = norm
            else:
                O, fX = self.oracle(self.Xk[k], U)

            self.fXk[k] = fX

            if np.linalg.norm(O) > self.grad_thresh:
                O = prev_grad
            else:
                prev_grad = O

            start = time.perf_counter()
            # Compute X_{k+1} (by projecting)
            self.Xk[k+1] = project_cone(self.Xk[k]-self.alpha[k]*O, d=0.000001, diag=True)
            end = time.perf_counter()
            self.time_proj += end-start

        # Compute error function for last iteration
        params = np.diag(self.Xk[self.N])

        # start = time.perf_counter()

        sol = self.sys.ocSolver(ini_state=self.x0, horizon=self.horizon, auxvar_value=params)

        # end = time.perf_counter()
        # self.time_sys += end-start


        if self.include_u:
            self.fXk[self.N] = self.loss_func(np.concatenate((sol['state_traj_opt'], np.append(sol['control_traj_opt'],np.zeros((1,sol['control_traj_opt'].shape[1])), axis=0)), axis=1))
        else:
            self.fXk[self.N] = self.loss_func(sol['state_traj_opt'])

        # Find optimal solution from lowest error function value
        i = np.argmin(self.fXk)
        self.X_sol = self.Xk[i]

        # Compute the state trajectory using optimal solution
        params = np.diag(self.X_sol)

        start = time.perf_counter()

        sol = self.sys.ocSolver(ini_state=self.x0, horizon=self.horizon, auxvar_value=params)

        end = time.perf_counter()
        self.time_sys += end-start

        self.x_sol_traj = sol['state_traj_opt']
        self.t_sol = sol['time']
        
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
                    ax_states[p][q].plot(self.t_sol, self.x_r.T[i], '--', color="tab:red", linewidth=1, label=r"$x^*$")
                    
                    ax_states[p][q].legend()
                else:
                    ax_states[p][q].plot(self.t_sol, self.x_sol_traj.T[i], "-", color="tab:blue", linewidth=2)
                    ax_states[p][q].plot(self.t_sol, self.x_r.T[i], '--', color="tab:red", linewidth=1)

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
                    ax_states[i].plot(self.t_sol, self.x_r.T[i], '--', color="tab:red", linewidth=1, label=r"$x^*$")
                    
                    ax_states[i].legend()
                else:
                    ax_states[i].plot(self.t_sol, self.x_sol_traj.T[i], "-", color="tab:blue", linewidth=2)
                    ax_states[i].plot(self.t_sol, self.x_r.T[i], '--', color="tab:red", linewidth=1)
                
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
        folder_name = "results/zorms/"+env
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        if plots:
            folder_name2 = "results/zorms/"+env+str(exp or '')
            if not os.path.exists(folder_name2):
                os.makedirs(folder_name2)
            # save figures
            self.fig_states.savefig(folder_name2+"/state.pdf",bbox_inches='tight')
            self.fig_error.savefig(folder_name2+"/error.pdf",bbox_inches='tight')

        if txt:
            folder_name2 = "results/zorms/"+env+str(exp or '')
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
            file.write(f"No. timesteps = {self.x_r.shape[0]} \n")
            file.write(f"T = {self.t_sol[-1]:0.2f} \n")
            file.write("\n")


            # algorithm parameters
            file.write("--- ZO-RMS parameters --- \n")
            file.write(f"N = {self.N}, ")
            file.write(f"mu = {self.mu:0.2e}, ")
            file.write(f"alpha = {self.alpha[0]:0.2e} \n")
            file.write("X0 = diag("+str(np.diag(self.Xk[0]))+") \n")
            file.write("\n")

            # results
            file.write("--- Results --- \n")
            file.write(f"f_X0 = {self.fXk[0]:0.4f}, f_min = {np.min(self.fXk):0.4f} \n")
            file.write(f"Percentage Improvement = {100*(self.fXk[0]-np.min(self.fXk))/self.fXk[0]:0.2f}% \n")
            file.write(f"Absolute Reduction = {self.fXk[0]-np.min(self.fXk):0.2f} \n")
            file.write("X* = diag("+str(np.diag(self.X_sol))+") \n")
            file.write(" \n")

            file.write("--- Timing --- \n")
            if self.time_total>120:
                file.write(f"Total time = {self.time_total/60:0.2f} mins \n")
            else:
                file.write(f"Total time = {self.time_total:0.2f}s \n")
            file.write(f"TOTAL:  GOE = {self.time_GOE:0.4f} s, sys = {self.time_sys:0.4f} s, proj = {self.time_proj:0.4f} s \n")
            file.write(f"AVERAGE:  GOE = {self.time_GOE/self.N:0.4f} s, sys = {self.time_sys/self.N:0.4f} s, proj = {self.time_proj/self.N:0.4f} s \n")
            file.write(f"PERCENTAGE:  GOE = {100*self.time_GOE/self.time_total:0.4f}%, sys = {100*self.time_sys/self.time_total:0.4f}%, proj = {100*self.time_proj/self.time_total:0.4f}%")

            if self.lipschitz_check:
                file.write(" \n")
                file.write(" \n")
                file.write("--- Lipschitz Stuff --- \n")
                file.write(f"L0 = {self.L0:0.4f}, r = {self.r:0.4f}")
            
            file.write(" \n")
            file.write(" \n")
            file.write("fXk = \n")
            file.write(str(self.fXk))

            file.close()

        if npz:
            # save stuff as numpy arrays
            if self.lipschitz_check:
                np.savez(folder_name+str(exp or '')+"_params",
                         error=self.fXk,
                         X0=np.diag(self.Xk[0]), 
                         X_sol=np.diag(self.X_sol),
                         X_true=true_X, 
                         alpha=self.alpha, 
                         mu=self.mu, 
                         time=self.time_total,
                         num_iter=self.N,
                         lipschitz_constant=self.L0,
                         bound=self.r)
            else:
                np.savez(folder_name+str(exp or '')+"_params", 
                         error=self.fXk, 
                         X0=np.diag(self.Xk[0]), 
                         X_sol=np.diag(self.X_sol), 
                         X_true=true_X, 
                         alpha=self.alpha, 
                         mu=self.mu, 
                         time=self.time_total,
                         num_iter=self.N)

    ''' Utility Methods '''

    def loss_func(self, x_i):
        # RMSE Error
        
        if self.RMSE:
            # use Root Mean Squared Error as loss function
            sum = 0
            for k in range(x_i.shape[0]):
                sum += np.linalg.norm(x_i[k]-self.x_r[k])**2
            
            sum = np.sqrt(sum/x_i.shape[0])
        else:
            # Use squared l2-norm as loss function
            sum = np.linalg.norm(x_i - self.x_r)**2

        return sum

    def oracle(self, X_i, U_i):

        # generate f(x_Xi)
        # get the parameter vector from diagonal
        params = np.diag(X_i) 

        start = time.perf_counter()

        sol = self.sys.ocSolver(ini_state=self.x0, horizon=self.horizon, auxvar_value=params)

        end = time.perf_counter()
        self.time_sys += end-start
        
        if self.include_u:
            fX = self.loss_func(np.concatenate((sol['state_traj_opt'], np.append(sol['control_traj_opt'],np.zeros((1,sol['control_traj_opt'].shape[1])), axis=0)), axis=1))
        else:
            fX = self.loss_func(sol['state_traj_opt'])

        # generate f(x_{Xi+mu*U})
        # get the parameter vector from diagonal
        params = np.diag(X_i + self.mu*U_i)

        start = time.perf_counter()

        sol = self.sys.ocSolver(ini_state=self.x0, horizon=self.horizon, auxvar_value=params)

        end = time.perf_counter()
        self.time_sys += end-start

        if self.include_u:
            fX_plus = self.loss_func(np.concatenate((sol['state_traj_opt'], np.append(sol['control_traj_opt'],np.zeros((1,sol['control_traj_opt'].shape[1])), axis=0)), axis=1))
        else:
            fX_plus = self.loss_func(sol['state_traj_opt'])

        O = (1/self.mu)*(fX_plus - fX)*U_i

        if not self.lipschitz_check:
            return O, fX
        else:
            X_plus = X_i + self.mu*U_i
            norm = np.linalg.norm(X_plus-X_i)
            return O, fX, fX_plus, norm
        
    def lipschitz_constant(self):
        if not self.lipschitz_check:
            return
        
        # r bound
        X0 = self.Xk[0]

        X_sol = self.X_sol

        self.r = np.linalg.norm(X0-X_sol)
        
        Xk = self.Xk[-1]

        # M = (self.fX_plus[0] - self.fXk[0])/self.norm[0]
        M = 0
        
        for i in range(self.N):
            # check if Lipschitz condition is violated
            if abs(self.fX_plus[i]-self.fXk[i]) >  M*self.norm[i]:
                # increase estimate
                M = abs(self.fX_plus[i] - self.fXk[i])/self.norm[i]

            Xi = self.Xk[i]

            # check if Lipschitz condition is violated
            if abs(self.fXk[-1]-self.fXk[i]) > M*np.linalg.norm(Xk-Xi):
                # increase estimate
                M = abs(self.fXk[-1] - self.fXk[i])/np.linalg.norm(Xk-Xi)

        self.L0 = M


class ZORMS_cont:
    def __init__(self, OC_sys, reference_traj, reference_time, X0, step_size, oracle_prec, num_iter, grad_protection_threshold = 1e5, lipschitz_check=False):
        
        ''' Reference Trajectory '''
        self.y_r = reference_traj
        self.t_points = reference_time

        ''' OC System '''
        # A system defined by COC_sys()
        self.sys = OC_sys
        # self.sys = Cont_COCSys()
        if not hasattr(self.sys, 'g'):
            self.sys.setMeasurement()

        # Details of parameters to be solved for
        self.m = self.sys.n_param

        ''' Algorithm Parameters '''
        # descent step-size
        self.alpha = step_size
        self.mu = oracle_prec
        self.N = num_iter

        # ignore steps where grad estimate is too large
        self.grad_thresh = grad_protection_threshold 

        ''' Iteration Data '''
        self.Xk = np.zeros((self.N+1, X0.shape[0], X0.shape[0]))
        self.Xk[0] = X0

        self.fXk = np.zeros(self.N+1)

        self.X_sol = None
        self.x_sol_traj = None
        self.t_sol = None
        
        ''' Timers '''
        self.time_total = 0
        self.time_GOE = 0
        self.time_sys = 0
        self.time_proj = 0

        self.fig_states = None
        self.fig_error = None

        # For Lipschitz test
        self.lipschitz_check = lipschitz_check
        if self.lipschitz_check:
            self.L0 = None
            self.norm = np.zeros(self.N+1)
            self.fX_plus = np.zeros(self.N+1)
            self.r = 0
    
    def run(self):
        start_tot = time.perf_counter()

        prev_grad = 0
            
        for k in tqdm(range(self.N)):
            start = time.perf_counter()
            # Generate Random Direction
            U = np.diag(np.random.standard_normal(self.m))
            end = time.perf_counter()
            self.time_GOE += end-start

            # Compute Oracle
            if self.lipschitz_check:
                O, fX, fX_plus, norm = self.oracle(self.Xk[k], U)
                self.fX_plus[k] = fX_plus
                self.norm[k] = norm
            else:
                O, fX = self.oracle(self.Xk[k], U)

            self.fXk[k] = fX

            if np.linalg.norm(O) > self.grad_thresh:
                O = prev_grad
            else:
                prev_grad = O

            start = time.perf_counter()
            # Compute X_{k+1} (by projecting)
            self.Xk[k+1] = project_cone(self.Xk[k]-self.alpha[k]*O, d=0.000001, diag=True)
            end = time.perf_counter()
            self.time_proj += end-start

        # Compute error function for last iteration
        params = np.diag(self.Xk[self.N])

        sol = self.sys.ocSolver(param_value=params)

        self.fXk[self.N] = self.error_func(sol)


        # Find optimal solution from lowest error function value
        i = np.argmin(self.fXk)
        self.X_sol = self.Xk[i]

        # Compute the state trajectory using optimal solution
        params = np.diag(self.X_sol)

        start = time.perf_counter()

        sol = self.sys.ocSolver(param_value=params)

        end = time.perf_counter()
        self.time_sys += end-start

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
        folder_name = "results/zorms/"+env
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        if plots:
            folder_name2 = "results/zorms/"+env+str(exp or '')
            if not os.path.exists(folder_name2):
                os.makedirs(folder_name2)
            # save figures
            self.fig_states.savefig(folder_name2+"/state.pdf",bbox_inches='tight')
            self.fig_error.savefig(folder_name2+"/error.pdf",bbox_inches='tight')

        if txt:
            folder_name2 = "results/zorms/"+env+str(exp or '')
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
            file.write("--- ZO-RMS parameters --- \n")
            file.write(f"N = {self.N}, ")
            file.write(f"mu = {self.mu:0.2e}, ")
            file.write(f"alpha = {self.alpha[0]:0.2e} \n")
            file.write("X0 = diag("+str(np.diag(self.Xk[0]))+") \n")
            file.write("\n")

            # results
            file.write("--- Results --- \n")
            file.write(f"f_X0 = {self.fXk[0]:0.4f}, f_min = {np.min(self.fXk):0.4f} \n")
            file.write(f"Percentage Improvement = {100*(self.fXk[0]-np.min(self.fXk))/self.fXk[0]:0.2f}% \n")
            file.write(f"Absolute Reduction = {self.fXk[0]-np.min(self.fXk):0.2f} \n")
            file.write("X* = diag("+str(np.diag(self.X_sol))+") \n")
            file.write(" \n")

            file.write("--- Timing --- \n")
            if self.time_total>120:
                file.write(f"Total time = {self.time_total/60:0.2f} mins \n")
            else:
                file.write(f"Total time = {self.time_total:0.2f}s \n")
            file.write(f"TOTAL:  GOE = {self.time_GOE:0.4f} s, sys = {self.time_sys:0.4f} s, proj = {self.time_proj:0.4f} s \n")
            file.write(f"AVERAGE:  GOE = {self.time_GOE/self.N:0.4f} s, sys = {self.time_sys/self.N:0.4f} s, proj = {self.time_proj/self.N:0.4f} s \n")
            file.write(f"PERCENTAGE:  GOE = {100*self.time_GOE/self.time_total:0.4f}%, sys = {100*self.time_sys/self.time_total:0.4f}%, proj = {100*self.time_proj/self.time_total:0.4f}%")

            if self.lipschitz_check:
                file.write(" \n")
                file.write(" \n")
                file.write("--- Lipschitz Stuff --- \n")
                file.write(f"L0 = {self.L0:0.4f}, r = {self.r:0.4f}")
            
            file.write(" \n")
            file.write(" \n")
            file.write("fXk = \n")
            file.write(str(self.fXk))

            file.close()

        if npz:
            # save stuff as numpy arrays
            if self.lipschitz_check:
                np.savez(folder_name+str(exp or '')+"_params",
                         error=self.fXk,
                         X0=np.diag(self.Xk[0]),
                         Xk = self.Xk, 
                         X_sol=np.diag(self.X_sol),
                         X_true=true_X, 
                         alpha=self.alpha, 
                         mu=self.mu, 
                         time=self.time_total,
                         num_iter=self.N,
                         lipschitz_constant=self.L0,
                         bound=self.r,
                         yr = self.y_r)
            else:
                np.savez(folder_name+str(exp or '')+"_params", 
                         error=self.fXk, 
                         X0=np.diag(self.Xk[0]), 
                         Xk = self.Xk,
                         X_sol=np.diag(self.X_sol), 
                         X_true=true_X, 
                         alpha=self.alpha, 
                         mu=self.mu, 
                         time=self.time_total,
                         num_iter=self.N,
                         yr = self.y_r)

    
    ''' Utility Methods '''
    def error_func(self, opt_sol):
        loss = 0

        for k, t in enumerate(self.t_points):
            y = self.sys.g_func(opt_sol['x_traj_opt'](t), opt_sol['u_traj_opt'](t)).full().flatten()

            loss += np.linalg.norm(self.y_r[k]-y)**2
        
        return loss

    def oracle(self, X_i, U_i):

        # generate f(x_Xi)
        # get the parameter vector from diagonal
        params = np.diag(X_i) 

        start = time.perf_counter()

        sol = self.sys.ocSolver(param_value=params)

        end = time.perf_counter()
        self.time_sys += end-start
        
        fX = self.error_func(sol)

        # generate f(x_{Xi+mu*U})
        # get the parameter vector from diagonal
        params = np.diag(X_i + self.mu*U_i)

        start = time.perf_counter()

        sol = self.sys.ocSolver(param_value=params)

        end = time.perf_counter()
        self.time_sys += end-start

        fX_plus = self.error_func(sol)

        O = (1/self.mu)*(fX_plus - fX)*U_i

        if not self.lipschitz_check:
            return O, fX
        else:
            X_plus = X_i + self.mu*U_i
            norm = np.linalg.norm(X_plus-X_i)
            return O, fX, fX_plus, norm
        
    def lipschitz_constant(self):
        if not self.lipschitz_check:
            return
        
        # r bound
        X0 = self.Xk[0]

        X_sol = self.X_sol

        self.r = np.linalg.norm(X0-X_sol)
        
        Xk = self.Xk[-1]

        # M = (self.fX_plus[0] - self.fXk[0])/self.norm[0]
        M = 0
        
        for i in range(self.N):
            # check if Lipschitz condition is violated
            if abs(self.fX_plus[i]-self.fXk[i]) >  M*self.norm[i]:
                # increase estimate
                M = abs(self.fX_plus[i] - self.fXk[i])/self.norm[i]

            Xi = self.Xk[i]

            # # check if Lipschitz condition is violated
            if abs(self.fXk[-1]-self.fXk[i]) > M*np.linalg.norm(Xk-Xi):
                # increase estimate
                M = abs(self.fXk[-1] - self.fXk[i])/np.linalg.norm(Xk-Xi)

        self.L0 = M
