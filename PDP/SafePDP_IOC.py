from PDP import SafePDP

from casadi import *
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
import os
import time


class SafePDP_IOC:
    def __init__(self, OC_sys, reference_traj, init_params, learning_rate, num_iter, gamma=1e-2, grad_protection_threshold = 1e5, strategy=np.array([1,2,3])):
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
        # differentiate
        self.sys.diffCPMP()
        # convert to unconstrained barrier OC
        self.sys.convert2BarrierOC(gamma=gamma)
        # Details of parameters to be solved for
        self.m = self.sys.n_auxvar

        ''' Algorithm Stuff '''
        self.lr = learning_rate
        self.N = num_iter
        self.clqr = SafePDP.EQCLQR()
        self.gamma = gamma

        self.grad_thresh = grad_protection_threshold 

        ''' Iteration Data '''
        # which strategies to use
        self.X0 = init_params
        if np.any(strategy==1):
            self.strat_1 = True

            self.Xk = np.zeros((self.N+1, init_params.shape[0]))
            self.Xk[0] = init_params
            self.loss = np.zeros(self.N+1)
            self.X_sol = None

            self.strat1_time = 0
        else:
            self.strat_1 = False
            self.loss = None

            self.strat1_time = None
        
        if np.any(strategy==2):
            self.strat_2 = True

            self.Xk_barrier1 = np.zeros((self.N+1, init_params.shape[0]))
            self.Xk_barrier1[0] = init_params
            self.loss_barrier1 = np.zeros(self.N+1)
            self.X_barrier1_sol = None

            self.strat2_time = 0

        else:
            self.strat_2 = False
            self.loss_barrier1 = None
            self.strat2_time = None

        if np.any(strategy==3):
            self.strat_3 = True

            self.Xk_barrier2 = np.zeros((self.N+1, init_params.shape[0]))
            self.Xk_barrier2[0] = init_params
            self.loss_barrier2 = np.zeros(self.N+1)
            self.X_barrier2_sol = None

            self.strat3_time = 0
        else:
            self.strat_3 = False
            self.loss_barrier2 = None
            self.strat3_time = None

        self.X_fin_sol = None
        self.x_sol_traj = None
        self.t_sol = None

        ''' Other stuff to save '''
        self.time_total = 0
        self.fig_states = None
        self.fig_error = None

    def run(self):
        start_tot = time.perf_counter()

        # Have to test 3 different things
        prev_grad = 0
        prev_grad_barrier1 = 0
        prev_grad_barrier2 = 0

        # fail flags
        strat1_fail = False
        strat2_fail = False
        strat3_fail = False

        for i in tqdm(range(self.N)):
            if self.strat_1 and not strat1_fail:
                try:
                    start = time.perf_counter()
                    ''' Strategy 1 '''
                    # use COC solver to compute traj and use Theorem 1 to compute derivative
                    
                    # learn current traj from current parameter guess
                    traj = self.sys.ocSolver(ini_state=self.x0, horizon=self.horizon, auxvar_value=self.Xk[i])

                    # establish auxiliary sys
                    aux_sys = self.sys.getAuxSys(
                        opt_sol=traj,
                        threshold=1e-1
                    )

                    # get the gradient and stuff
                    self.clqr.auxsys2Eqctlqr(auxsys=aux_sys)
                    aux_sol = self.clqr.eqctlqrSolver(threshold=1e-1)
                    self.loss[i], grad = self.Traj_L2_Loss(traj, aux_sol)

                    # protect non-differentiable case
                    if np.linalg.norm(grad) > self.grad_thresh:
                        grad = prev_grad
                    else:
                        prev_grad = grad
                    
                    # update
                    self.Xk[i+1] = self.Xk[i] - self.lr[i]*grad

                    self.strat1_time += time.perf_counter() - start
                except:
                    print("STRATEGY 1 FAILED")
                    strat1_fail = True
                    if i == 0:
                        traj = self.sys.ocSolver(ini_state=self.x0, horizon=self.horizon, auxvar_value=self.Xk[i])
                        if self.include_u:
                            x_traj = np.concatenate((traj['state_traj_opt'], np.append(traj['control_traj_opt'],np.zeros((1,traj['control_traj_opt'].shape[1])), axis=0)), axis=1)
                        else:
                            x_traj = traj['state_traj_opt']
                        self.loss[i:] = np.linalg.norm(x_traj - self.xr)**2
                        self.Xk[i:] = self.Xk[i]
                    else:
                        self.loss[i:] = self.loss[i-1]
                        self.Xk[i:] = self.Xk[i-1]
                    self.strat1_time = np.nan

            if self.strat_2 and not strat2_fail:
                try:
                    start = time.perf_counter()
                    ''' Strategy 2 '''
                    # use Theorem 2 to approximate both the system trajectory and its derivative

                    # get trajectory
                    traj_barrier1 = self.sys.solveBarrierOC(
                        horizon=self.horizon,
                        ini_state = self.x0,
                        auxvar_value = self.Xk_barrier1[i]
                    )

                    # get aux system solution
                    aux_sol_barrier1 = self.sys.auxSysBarrierOC(opt_sol=traj_barrier1)
                    
                    # get loss and gradient
                    self.loss_barrier1[i], grad_barrier1 = self.Traj_L2_Loss(traj_barrier1, aux_sol_barrier1)

                    # protect non-differentiable case
                    if np.linalg.norm(grad_barrier1) > self.grad_thresh:
                        grad_barrier1 = prev_grad_barrier1
                    else:
                        prev_grad_barrier1 = grad_barrier1
                    
                    # update
                    self.Xk_barrier1[i+1] = self.Xk_barrier1[i] - self.lr[i]*grad_barrier1

                    self.strat2_time += time.perf_counter() - start
                except:
                    print("STRATEGY 2 FAILED")
                    strat2_fail = True
                    if i == 0:
                        traj = self.sys.ocSolver(ini_state=self.x0, horizon=self.horizon, auxvar_value=self.Xk_barrier1[i])
                        if self.include_u:
                            x_traj = np.concatenate((traj['state_traj_opt'], np.append(traj['control_traj_opt'],np.zeros((1,traj['control_traj_opt'].shape[1])), axis=0)), axis=1)
                        else:
                            x_traj = traj['state_traj_opt']
                        self.loss_barrier1[i:] = np.linalg.norm(x_traj - self.xr)**2
                        self.Xk_barrier1[i:] = self.Xk_barrier1[i]
                    else:
                        self.loss_barrier1[i:] = self.loss_barrier1[i-1]
                        self.Xk_barrier1[i:] = self.Xk_barrier1[i-1]
                    self.strat2_time = np.nan
            
            if self.strat_3 and not strat3_fail:
                try:
                    start = time.perf_counter()
                    ''' Strategy 3 '''
                    # use COC solver to compute trajectory and Theorem 2 to approximate the trajectory derivative

                    # get trajectories
                    traj_COC = self.sys.ocSolver(ini_state=self.x0, horizon=self.horizon, auxvar_value=self.Xk_barrier2[i])
                    traj_barrier2 = self.sys.solveBarrierOC(
                        horizon=self.horizon,
                        ini_state = self.x0,
                        auxvar_value = self.Xk_barrier2[i]
                    )

                    # get aux sys solution
                    aux_sol_barrier2 = self.sys.auxSysBarrierOC(opt_sol=traj_barrier2)

                    # get loss and gradient
                    self.loss_barrier2[i], grad_barrier2 = self.Traj_L2_Loss(traj_COC, aux_sol_barrier2)

                    # protect non-differentiable case
                    if np.linalg.norm(grad_barrier2) > self.grad_thresh:
                        grad_barrier2 = prev_grad_barrier2
                    else:
                        prev_grad_barrier1 = grad_barrier2
                    
                    # update
                    self.Xk_barrier2[i+1] = self.Xk_barrier2[i] - self.lr[i]*grad_barrier2

                    self.strat3_time += time.perf_counter() - start
                except:
                    print("STRATEGY 3 FAILED")
                    strat3_fail = True
                    if i == 0:
                        traj = self.sys.ocSolver(ini_state=self.x0, horizon=self.horizon, auxvar_value=self.Xk_barrier2[i])
                        if self.include_u:
                            x_traj = np.concatenate((traj['state_traj_opt'], np.append(traj['control_traj_opt'],np.zeros((1,traj['control_traj_opt'].shape[1])), axis=0)), axis=1)
                        else:
                            x_traj = traj['state_traj_opt']
                        self.loss_barrier2[i:] = np.linalg.norm(x_traj - self.xr)**2
                        self.Xk_barrier2[i:] = self.Xk_barrier2[i]
                    else:
                        self.loss_barrier2[i:] = self.loss_barrier2[i-1]
                        self.Xk_barrier2[i:] = self.Xk_barrier2[i-1]
                    self.strat3_time = np.nan

        # compute loss for final iteration and solution
        mins = []
        sols = []
        if self.strat_1:
            ''' Strategy 1 '''
            start = time.perf_counter()

            traj = self.sys.ocSolver(ini_state=self.x0, horizon=self.horizon, auxvar_value=self.Xk[self.N])
            if self.include_u:
                x_traj = np.concatenate((traj['state_traj_opt'], np.append(traj['control_traj_opt'],np.zeros((1,traj['control_traj_opt'].shape[1])), axis=0)), axis=1)
            else:
                x_traj = traj['state_traj_opt']
            self.loss[self.N] = np.linalg.norm(x_traj - self.xr)**2

            self.X_sol = self.Xk[np.argmin(self.loss)]
            mins += [np.min(self.loss)]
            sols.append(self.X_sol)

            self.strat1_time += time.perf_counter() - start

        if self.strat_2:
            ''' Strategy 2 '''
            start = time.perf_counter()

            traj = self.sys.solveBarrierOC(ini_state=self.x0, horizon=self.horizon, auxvar_value=self.Xk_barrier1[self.N])
            if self.include_u:
                x_traj = np.concatenate((traj['state_traj_opt'], np.append(traj['control_traj_opt'],np.zeros((1,traj['control_traj_opt'].shape[1])), axis=0)), axis=1)
            else:
                x_traj = traj['state_traj_opt']
            self.loss_barrier1[self.N] = np.linalg.norm(x_traj - self.xr)**2

            self.X_barrier1_sol = self.Xk_barrier1[np.argmin(self.loss_barrier1)]
            mins += [np.min(self.loss_barrier1)]
            sols.append(self.X_barrier1_sol)

            self.strat2_time += time.perf_counter() - start

        if self.strat_3:
            ''' Strategy 3 '''
            start = time.perf_counter()

            traj = self.sys.ocSolver(ini_state=self.x0, horizon=self.horizon, auxvar_value=self.Xk_barrier2[self.N])
            if self.include_u:
                x_traj = np.concatenate((traj['state_traj_opt'], np.append(traj['control_traj_opt'],np.zeros((1,traj['control_traj_opt'].shape[1])), axis=0)), axis=1)
            else:
                x_traj = traj['state_traj_opt']
            self.loss_barrier2[self.N] = np.linalg.norm(x_traj - self.xr)**2

            self.X_barrier2_sol = self.Xk_barrier2[np.argmin(self.loss_barrier2)]
            mins += [np.min(self.loss_barrier2)]
            sols.append(self.X_barrier2_sol)

            self.strat3_time += time.perf_counter() - start

        # Find optimal solution from lowest error function value
        self.X_fin_sol = sols[np.argmin(mins)]

        # Compute the state trajectory using optimal solution
        sol = self.sys.ocSolver(ini_state=self.x0, horizon=self.horizon, auxvar_value=self.X_fin_sol)
        self.x_sol_traj = sol['state_traj_opt']
        self.t_sol = sol['time']

        end_tot = time.perf_counter()
        self.time_total = end_tot - start_tot
    
    def Traj_L2_Loss(self, traj, aux_sol):
        demo_state_traj = self.xr[:,:self.sys.n_state]
        if self.include_u:
                demo_control_traj = self.xr[:self.horizon,self.sys.n_state:]
            
        state_traj = traj['state_traj_opt']
        control_traj = traj['control_traj_opt']

        dldx_traj = 2*(state_traj - demo_state_traj)
        if self.include_u:
            dldu_traj = 2*(control_traj - demo_control_traj)
            loss = numpy.linalg.norm(0.5*dldx_traj) ** 2 + numpy.linalg.norm(0.5*dldu_traj) ** 2
        else:
            loss = numpy.linalg.norm(0.5*dldx_traj) ** 2

        dxdp_traj = aux_sol['state_traj_opt']
        dudp_traj = aux_sol['control_traj_opt']

        # use chain rule to compute the gradient
        dl = 0
        for t in range(self.horizon):
            if self.include_u:
                dl = dl + np.matmul(dldx_traj[t, :], dxdp_traj[t]) + np.matmul(dldu_traj[t, :], dudp_traj[t])
            else:
                dl = dl + np.matmul(dldx_traj[t, :], dxdp_traj[t])
        dl = dl + numpy.dot(dldx_traj[-1, :], dxdp_traj[-1])

        return loss, dl

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
        if self.strat_1:
            ax_error.plot(np.arange(0,self.loss.shape[0]), self.loss, label="Strategy 1")
        if self.strat_2:
            ax_error.plot(np.arange(0,self.loss_barrier1.shape[0]), self.loss_barrier1, label="Strategy 2")
        if self.strat_3:
            ax_error.plot(np.arange(0,self.loss_barrier2.shape[0]), self.loss_barrier2, label="Strategy 3")
        ax_error.set_ylabel(r"$f(X_k)$")
        ax_error.set_xlabel(r"$k$")
        ax_error.legend()

        if self.strat_1:
            ax_error.text(0.01, 0.05, f'{(self.loss[0]-np.min(self.loss)):0.2f} Absolute', horizontalalignment='left', verticalalignment='bottom', transform=ax_error.transAxes)
            ax_error.text(0.01, 0.05, f'{100*(self.loss[0]-np.min(self.loss))/self.loss[0]:0.2f}\%  Improvement', horizontalalignment='left', verticalalignment='top', transform=ax_error.transAxes)
        elif self.strat_2:
            ax_error.text(0.01, 0.05, f'{(self.loss_barrier1[0]-np.min(self.loss_barrier1)):0.2f} Absolute', horizontalalignment='left', verticalalignment='bottom', transform=ax_error.transAxes)
            ax_error.text(0.01, 0.05, f'{100*(self.loss_barrier1[0]-np.min(self.loss_barrier1))/self.loss[0]:0.2f}\%  Improvement', horizontalalignment='left', verticalalignment='top', transform=ax_error.transAxes)
        elif self.strat_3:
            ax_error.text(0.01, 0.05, f'{(self.loss_barrier2[0]-np.min(self.loss_barrier2)):0.2f} Absolute', horizontalalignment='left', verticalalignment='bottom', transform=ax_error.transAxes)
            ax_error.text(0.01, 0.05, f'{100*(self.loss_barrier2[0]-np.min(self.loss_barrier2))/self.loss[0]:0.2f}\%  Improvement', horizontalalignment='left', verticalalignment='top', transform=ax_error.transAxes)
    
    def save_results(self, env, exp=None, true_X=None, plots=False, txt=False, npz=False):
        # create a folder to save the results
        folder_name = "results/safe_pdp/"+env
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        if plots:
            folder_name2 = "results/safe_pdp/"+env+str(exp or '')
            if not os.path.exists(folder_name2):
                os.makedirs(folder_name2)
            # save figures
            self.fig_states.savefig(folder_name2+"/state.pdf",bbox_inches='tight')
            self.fig_error.savefig(folder_name2+"/error.pdf",bbox_inches='tight')

        if txt:
            folder_name2 = "results/safe_pdp/"+env+str(exp or '')
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
            file.write(f"Learning rate = {self.lr[0]:0.2e} \n ")
            file.write("X0 = "+str(self.Xk[0])+" \n")
            file.write(f"Barrier OC gamma = {self.gamma:0.2e} \n")
            file.write(f"Gradient Protection Threshold = {self.grad_thresh:0.2e}")
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
            if self.strat_1:
                file.write(f"Strategy 1: {self.strat1_time:0.2e}s \n")
            if self.strat_2:
                file.write(f"Strategy 2: {self.strat2_time:0.2e}s \n")
            if self.strat_3:
                file.write(f"Strategy 3: {self.strat3_time:0.2e}s \n")
            
            # file.write(" \n")
            # file.write(" \n")
            # file.write("loss = \n")
            # file.write(str(self.loss))

            file.close()

        if npz:
            # save stuff as numpy arrays
            np.savez(folder_name+str(exp or '')+"_params", 
                     error=self.loss, 
                     error_barrier1=self.loss_barrier1, 
                     error_barrier2=self.loss_barrier2, 
                     X0=self.X0, 
                     X_sol=self.X_fin_sol,
                     X_true=true_X,
                     learning_rate=self.lr,
                     time1=self.strat1_time,
                     time2=self.strat2_time,
                     time3=self.strat3_time,
                     num_iter=self.N)
    