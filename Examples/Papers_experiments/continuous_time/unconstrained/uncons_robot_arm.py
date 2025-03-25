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
    env = "uncons_robot_arm"
    
    save = False

    # init system environment
    arm = JinEnv.RobotArm()
    l1, m1, l2, m2 = 1, 1, 1, 1
    arm.initDyn(g=0)

    wq1, wq2, wdq1, wdq2 = 1, 1, 1, 1
    arm.initCost(wu=0.5)

    # system parameters
    dt = 0.1
    n_points = 35
    horizon = dt*n_points 

    ini_state = [-np.pi/2, 3*np.pi/4, 0, 0]

    # OC solver (PDP)
    sys_pdp = CPDP.COCSys()
    sys_pdp.setAuxvarVariable(vertcat(arm.dyn_auxvar, arm.cost_auxvar))
    sys_pdp.setStateVariable(arm.X)
    sys_pdp.setControlVariable(arm.U)
    dyn = arm.f
    sys_pdp.setDyn(dyn)
    sys_pdp.setPathCost(arm.path_cost)   
    sys_pdp.setFinalCost(arm.final_cost)
    sys_pdp.setIntegrator(n_grid=n_points)

    g = sys_pdp.state[0:2]

    # OC solver (ZORMS)
    sys_zorms = Cont_OCSys()
    sys_zorms.setParamVariable(vertcat(arm.dyn_auxvar, arm.cost_auxvar))
    sys_zorms.setStateVariable(arm.X)
    sys_zorms.setInputVariable(arm.U)
    dyn = arm.f
    sys_zorms.setDynamics(dyn)
    sys_zorms.setStageCost(arm.path_cost)   
    sys_zorms.setTerminalCost(arm.final_cost)

    sys_zorms.setInitialCondition(ini_state)
    sys_zorms.setHorizon(horizon, n_points)

    sys_zorms.setMeasurement(sys_zorms.x[0:2])

    # true parameters
    param_mean = np.array([l1, m1, l2, m2, wq1, wq2, wdq1, wdq2])
    param_sigma = param_mean/(param_mean+4)

    rng = np.random.default_rng()
    
    true_param = rng.normal(param_mean, param_sigma)
    while np.any(true_param<=0):
        true_param = rng.normal(param_mean, param_sigma)
    
    # reference trajectory
    sol = sys_zorms.ocSolver(true_param)

    x = sol['x_traj_opt']

    t = np.linspace(0, horizon, 10)
    yr = x(t)[:, :2]
    

    for j in trial_num:
    
        true_param = rng.normal(param_mean, param_sigma)
        while np.any(true_param<=0):
            true_param = rng.normal(param_mean, param_sigma)
        
        # reference trajectory
        sol = sys_zorms.ocSolver(true_param)

        x = sol['x_traj_opt']

        t = np.linspace(0, horizon, 10)
        yr = x(t)[:, :2]
        
        for i in num_trials:

            sigma = 0.9
            X0 = true_param + sigma*np.random.random(len(true_param))

            # iterations
            N = 50
            
            # zorms stuff
            mu = 5 * 10**(-4)
            alpha = 5 * 10**(-3) * np.ones(N)

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