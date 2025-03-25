from PDP import PDP
from JinEnv import JinEnv
from ZORMS_LfD.ZORMS_LfD import ZORMS_disc
from PDP.PDP_IOC import PDP_IOC
from IDOC.IDOC_IOC import IDOC_uncons

import numpy as np
import matplotlib.pyplot as plt
from casadi import *

import time

if __name__ == "__main__":
    trial_num = range(5)
    num_trials = range(5)
    env = "uncons_cartpole"
    input = True
    
    save = False

    # init system environment
    cartpole = JinEnv.CartPole()
    mc, mp, l = 0.5, 0.5, 1
    cartpole.initDyn() 
    
    wx, wq, wdx, wdq = 1, 1, 6, 1
    cartpole.initCost()

    # initial parameters
    dt = 0.1
    # horizon = int(5/dt) # N = seconds / dt
    horizon = 30
    ini_state = [0, 0, 0, 0]
    # ini_state = [0, -0.5, 0, 0]
    # ini_state = [0, -0.25, 0, 0]
    # ini_state = [0, 0.5, 0, 0]
    # ini_state = [0, 0.25, 0, 0]

    # OC solver
    sys = PDP.OCSys()
    sys.setAuxvarVariable(vertcat(cartpole.dyn_auxvar, cartpole.cost_auxvar))
    sys.setStateVariable(cartpole.X)
    sys.setControlVariable(cartpole.U)
    dyn = cartpole.X + dt * cartpole.f
    sys.setDyn(dyn)
    sys.setPathCost(cartpole.path_cost)   
    sys.setFinalCost(cartpole.final_cost)

    # true parameters
    param_mean = np.array([mc,mp,l,wx, wq, wdx, wdq])
    param_sigma = param_mean/(param_mean+5)

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

            # zorms stuff
            if not input:
                ## w/o input traj ##
                mu = 5.7 * 10**(-4)
                # alpha = 3.45 * 10**(-2) /np.sqrt(np.arange(0,N)+1)
                alpha = 3 * 10**(-2) * np.ones(N)

                expr = "/trial"+str(trial_num)+"_noinput"+str(i)
            else:
                ## w/ input traj ##
                mu = 5.7 * 10**(-4)
                # alpha = 6.901 * 10**(-3) /np.sqrt(np.arange(0,N)+1)
                alpha = 3 * 10**(-3) * np.ones(N)

                expr = "/trial"+str(trial_num)+"_input"+str(i)
            

            zorms = ZORMS_disc(sys, xr, np.diag(X0), alpha, mu, N, lipschitz_check=True)

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

            pdp = PDP_IOC(sys, xr, X0, lr, N)

            start = time.perf_counter()
            pdp.run()
            end = time.perf_counter()
            print(f"PDP finished in {end - start:0.2f} seconds")
            
            # pdp.plot_results()
            if save:
                pdp.save_results(env, exp=expr, true_X=true_param, npz=True)

            # idoc stuff
            idoc = IDOC_uncons(sys, xr, X0, lr, N)

            start = time.perf_counter()
            idoc.run()
            end = time.perf_counter()
            print(f"IDOC finished in {end - start:0.2f} seconds")
            
            # idoc.plot_results()
            if save:
                idoc.save_results(env, exp=expr, true_X=true_param, npz=True)


            # plt.show()