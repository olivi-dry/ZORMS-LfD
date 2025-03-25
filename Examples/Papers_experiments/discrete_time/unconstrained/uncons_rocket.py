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
    env = "uncons_rocket"
    input = True
    
    save = False

    # init system environment
    rocket = JinEnv.Rocket()
    Jx, Jy, Jz, m, l = 0.5, 1, 1, 1, 1
    rocket.initDyn() 
    
    wr, wv, wtilt, ww, wsthrust = 10, 1, 50, 1, 1
    rocket.initCost(wthrust=0.4)

    # trajectory parameters
    dt = 0.1
    horizon = 40

    ini_r_I=[10, -8, 5.]
    # ini_v_I = [-.5, 0.0, 0.0]
    ini_v_I = [-.1, 0.0, -0.0]
    # ini_q = JinEnv.toQuaternion(1.,[0,1,-1])
    ini_q = JinEnv.toQuaternion(0,[1, 0, 0])
    # ini_w = [0, -0.0, 0.0]
    ini_w = [0, 0, 0]
    ini_state = ini_r_I + ini_v_I + ini_q + ini_w

    # OC solver
    sys = PDP.OCSys()
    sys.setAuxvarVariable(vertcat(rocket.dyn_auxvar, rocket.cost_auxvar))
    sys.setStateVariable(rocket.X)
    sys.setControlVariable(rocket.U)
    dyn = rocket.X + dt * rocket.f
    sys.setDyn(dyn)
    sys.setPathCost(rocket.path_cost)   
    sys.setFinalCost(rocket.final_cost)

    # true parameters
    param_mean = np.array([Jx, Jy, Jz, m, l, wr, wv, wtilt, ww, wsthrust])
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

            sigma = 0.4
            X0 = true_param + sigma*np.random.random(len(true_param))

            # iterations
            N = 100

            # zorms stuff
            if not input:
                ## w/o input traj ##
                mu = 5.4 * 10**(-5)
                # alpha = 2.904 * 10**(-4) /np.sqrt(np.arange(0,N)+1)
                alpha = 1 * 10**(-4) * np.ones(N)

                expr = "/trial"+str(trial_num)+"_noinput"+str(i)
            else:
                ## w/ input traj ##
                mu = 3.37 * 10**(-6)
                # alpha = 2.904 * 10**(-4) /np.sqrt(np.arange(0,N)+1)
                alpha = 1 * 10**(-5) * np.ones(N)

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