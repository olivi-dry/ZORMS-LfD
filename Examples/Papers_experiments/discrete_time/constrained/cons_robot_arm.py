from PDP import SafePDP
from JinEnv import JinEnv
from ZORMS_LfD.ZORMS_LfD import ZORMS_disc
from PDP.SafePDP_IOC import SafePDP_IOC
from IDOC.IDOC_IOC import IDOC


import numpy as np
import matplotlib.pyplot as plt
from casadi import *

import time

if __name__ == "__main__":
    trial_num = range(5)
    num_trials = range(5)
    env = "cons_robot_arm"
    input = True
    
    save = False

    # init system environment
    arm = JinEnv.RobotArm()
    l1, m1, l2, m2 = 1, 1, 1, 1
    arm.initDyn(g=0)

    wq1, wq2, wdq1, wdq2 = 1, 1, 0.5, 0.5
    arm.initCost()

    max_u, max_q = 1, pi
    arm.initConstraints()

    # trajectory parameters
    dt = 0.1
    horizon = 35

    ini_state = [-np.pi/2, 3*np.pi/4, 0, 0]

    # OC solver
    sys = SafePDP.COCsys()
    sys.setAuxvarVariable(vertcat(arm.dyn_auxvar, arm.constraint_auxvar, arm.cost_auxvar))
    sys.setStateVariable(arm.X)
    sys.setControlVariable(arm.U)
    dyn = arm.X + dt * arm.f
    sys.setDyn(dyn)
    sys.setPathCost(arm.path_cost)   
    sys.setFinalCost(arm.final_cost)
    sys.setPathInequCstr(arm.path_inequ)

    # true parameters
    param_mean = np.array([l1, m1, l2, m2, max_u, max_q, wq1, wq2, wdq1, wdq2])
    param_sigma = param_mean/(param_mean+4)

    rng = np.random.default_rng()

    for j in trial_num:
        true_param = rng.normal(param_mean, param_sigma)
        while np.any(true_param<=0):
            true_param = rng.normal(param_mean, param_sigma)
        
        # reference trajectory
        sol = sys.ocSolver(ini_state=ini_state, horizon=horizon, auxvar_value=true_param)
        if not input:
            ## w/o input traj ##
            xr = sol['state_traj_opt']
        else:
            ## w/ input traj ##
            xr = np.concatenate((sol['state_traj_opt'], np.append(sol['control_traj_opt'],np.zeros((1,sol['control_traj_opt'].shape[1])), axis=0)), axis=1)

        for i in num_trials:

            sigma = 0.9
            X0 = true_param + sigma*np.random.random(len(true_param))

            # iterations
            N = 100
            thresh = 1e5

            # zorms stuff
            if not input:
                ## w/o input traj ##
                mu = 1.66 * 10**(-4)
                alpha = 1.081 * 10**(-3) * np.ones(N)

                expr = "/trial"+str(j)+"_noinput"+str(i)
            else:
                ## w/ input traj ##
                mu = 5 * 10**(-5)
                alpha = 5.4 * 10**(-4) * np.ones(N)

                expr = "/trial"+str(j)+"_input"+str(i)
            

            zorms = ZORMS_disc(sys, xr, np.diag(X0), alpha, mu, N, lipschitz_check=True, grad_protection_threshold=thresh)

            start = time.perf_counter()

            zorms.run()

            end = time.perf_counter()
            print(f"ZORMS finished in {end - start:0.2f} seconds")

            # zorms.plot_results()
            zorms.lipschitz_constant()
            
            if save:
                zorms.save_results(env, exp=expr, true_X=true_param, npz=True)

            # pdp stuff
            lr = alpha
            gamma = 1e-2
            # thresh = 1e5

            safe_pdp = SafePDP_IOC(sys, xr, X0, lr, N, gamma=gamma, grad_protection_threshold=thresh)

            start = time.perf_counter()

            safe_pdp.run()

            end = time.perf_counter()
            print(f"SafePDP finished in {end - start:0.2f} seconds")
            
            # safe_pdp.plot_results()
            if save:
                safe_pdp.save_results(env, exp=expr, true_X=true_param, npz=True)

            # idoc stuff
            idoc = IDOC(sys, xr, X0, lr, N, gamma=gamma, grad_protection_threshold=thresh, strategy=1)

            start = time.perf_counter()

            idoc.run()

            end = time.perf_counter()
            print(f"IDOC finished in {end - start:0.2f} seconds")
            
            # idoc.plot_results()
            if save:
                idoc.save_results(env, exp=expr, true_X=true_param, npz=True)

            # plt.show()