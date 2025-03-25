from PDP import CPDP
from JinEnv import JinEnv
from ZORMS_LfD.COC_Sys import Cont_OCSys

from ZORMS_LfD.ZORMS_LfD import ZORMS_cont
from PDP.CPDP_IOC import CPDP_IOC


import numpy as np
import matplotlib.pyplot as plt
from casadi import *

import time

if __name__ == "__main__":
    trial_num = range(5)
    num_trials = range(5)
    env = "uncons_quadrotor"
    
    save = False

    # init system environment
    quad = JinEnv.Quadrotor()
    Jx, Jy, Jz, m, l = 1, 1, 1, 1, 1
    quad.initDyn(c=0.02) 
    
    wr, wv, wq, ww = 1, 1, 1, 1
    quad.initCost()

    # initial parameters
    dt = 0.1
    n_points = 25
    horizon = dt*50

    alt_r_I=[[-8, -6, 9.],[8, 6, 9.]]
    ini_v_I = [0.0, 0.0, 0.0]
    ini_q = JinEnv.toQuaternion(0,[1,-1,1])
    ini_w = [0.0, 0.0, 0.0]

    ini_state = alt_r_I[0] + ini_v_I + ini_q + ini_w
    # ini_state = alt_r_I[1] + ini_v_I + ini_q + ini_w

    # OC solver
    sys_pdp = CPDP.COCSys()
    sys_pdp.setAuxvarVariable(vertcat(quad.dyn_auxvar, quad.cost_auxvar))
    sys_pdp.setStateVariable(quad.X)
    sys_pdp.setControlVariable(quad.U)
    dyn = quad.f
    sys_pdp.setDyn(dyn)
    sys_pdp.setPathCost(quad.path_cost)   
    sys_pdp.setFinalCost(quad.final_cost)
    sys_pdp.setIntegrator(n_grid=n_points)

    g = vertcat(sys_pdp.state[0:3], sys_pdp.state[6:10])

    # OC solver (ZORMS)
    sys_zorms = Cont_OCSys()
    sys_zorms.setParamVariable(vertcat(quad.dyn_auxvar, quad.cost_auxvar))
    sys_zorms.setStateVariable(quad.X)
    sys_zorms.setInputVariable(quad.U)
    dyn = quad.f
    sys_zorms.setDynamics(dyn)
    sys_zorms.setStageCost(quad.path_cost)   
    sys_zorms.setTerminalCost(quad.final_cost)

    sys_zorms.setInitialCondition(ini_state)
    sys_zorms.setHorizon(horizon, n_points)

    sys_zorms.setMeasurement(vertcat(sys_zorms.x[0:3], sys_zorms.x[6:10]))

    # true parameters
    param_mean = np.array([Jx, Jy, Jz, m, l, wr, wv, wq, ww])
    param_sigma = param_mean/(param_mean+4)

    rng = np.random.default_rng()
    
    for j in trial_num:
    
        true_param = rng.normal(param_mean, param_sigma)
        while np.any(true_param<=0):
            true_param = rng.normal(param_mean, param_sigma)
        
        # reference trajectory
        sol = sys_zorms.ocSolver(true_param)

        x = sol['x_traj_opt']

        t = np.linspace(0, horizon, 10)
        yr = x(t)[:, np.array([0,1,2,6,7,8,9])]
        
        for i in num_trials:

            sigma = 0.9
            X0 = true_param + sigma*np.random.random(len(true_param))

            # iterations
            N = 50
            
            # zorms stuff
            mu = 8 * 10**(-4)
            alpha = 6 * 10**(-3) * np.ones(N)

            expr = "/trial"+str(j)+"_"+str(i)

            zorms = ZORMS_cont(sys_zorms, yr, t, np.diag(X0), alpha, mu, N, lipschitz_check=True)

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

            pdp = CPDP_IOC(sys_pdp, yr, t, ini_state, horizon, X0, lr, N, measurement_func=g)

            start = time.perf_counter()
            pdp.run()
            end = time.perf_counter()
            print(f"PDP finished in {end - start:0.2f} seconds")
            
            # pdp.plot_results()
            
            if save:
                pdp.save_results(env, exp=expr, true_X=true_param, npz=True)

            # plt.show()