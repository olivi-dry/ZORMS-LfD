import casadi as ca
import numpy as np

from scipy.interpolate import CubicSpline

class Cont_OCSys:
    """
    
    - define a Continuous Optimal Control System
        - state and input variables
        - dynamics (ODE)
        - terminal and stage cost functions
        - parameters (unknown)
    - solve the OC system to obtain the state and input trajectory

    """    
    def __init__(self):
        # self.inf = ca.inf
        self.inf = 1e20

    def setStateVariable(self, state, lower_bound=[], upper_bound=[]):
        self.x = state # match notation
        self.n_state = self.x.numel() # this is 'n'

        if len(lower_bound) == self.n_state:
            self.state_lb = lower_bound
        else:
            self.state_lb = self.n_state * [-self.inf]

        if len(upper_bound) == self.n_state:
            self.state_ub = upper_bound
        else:
            self.state_ub = self.n_state * [self.inf]
    
    def setInputVariable(self, input, lower_bound=[], upper_bound=[]):
        self.u = input # match notation
        self.n_input = self.u.numel() # this is 'm'

        if len(lower_bound) == self.n_input:
            self.input_lb = lower_bound
        else:
            self.input_lb = self.n_input * [-self.inf]

        if len(upper_bound) == self.n_input:
            self.input_ub = upper_bound
        else:
            self.input_ub = self.n_input * [self.inf]

    def setParamVariable(self, param=None):
        # create useless variable if no unknown parameters
        if param is None or param.numel() == 0:
            self.theta = ca.SX.sym('theta')
        else:
            self.theta = param
        self.n_param = self.theta.numel()
    
    def setDynamics(self, ODE):
        # want dynamics as an ODE
        # dx/dt = f(x,u) 

        if not hasattr(self, 'theta'):
            self.setParamVariable()
        
        # keep symbolic expression (of LHS dynamics)
        self.f = ODE
        # create CasADI function object
        self.f_func = ca.Function('dynamics', [self.x, self.u, self.theta], [self.f])

    def setMeasurement(self, measurement_func=None):
        # y = g(x,u)
        if not hasattr(self, 'theta'):
            self.setParamVariable()
        
        if measurement_func is None:
            self.g = ca.vertcat(self.x, self.u)
        else:
            self.g = measurement_func
        
        self.g_func = ca.Function('measurements', [self.x, self.u], [self.g])
    
    def setStageCost(self, stage_cost):
        if not hasattr(self, 'theta'):
            self.setParamVariable()
        
        assert stage_cost.numel() == 1, "stage_cost must be a scalar function"

        # keep symbolic expression
        self.l = stage_cost
        # create CasADI function object
        self.l_func = ca.Function('stage_cost', [self.x, self.u, self.theta], [self.l])
    
    def setTerminalCost(self, term_cost):
        if not hasattr(self, 'theta'):
            self.setParamVariable()
        
        assert term_cost.numel() == 1, "term_cost must be a scalar function"

        # keep symbolic expression
        self.h = term_cost
        # create CasADI function object
        self.h_func = ca.Function('terminal_cost', [self.x, self.theta], [self.h])

    def setInitialCondition(self, initial_state=None):
        self.x0 = initial_state

    def setHorizon(self, horizon=1, num_traj_points=20):
        self.T = horizon
        self.N = num_traj_points

    def ocSolver2(self, param_value=1, initial_state=None, time_horizon=1, print_level=0):
        # Method is direct collocation
        
        assert hasattr(self, 'x'), "Define the state variable first!"
        assert hasattr(self, 'u'), "Define the control variable first!"
        assert hasattr(self, 'f'), "Define the system dynamics first!"
        assert hasattr(self, 'l'), "Define the state cost first!"
        assert hasattr(self, 'h'), "Define the terminal cost first!"
        
        if not hasattr(self, 'x0'):
            self.setInitialCondition(initial_state)
        if not hasattr(self, 'T'):
            self.setHorizon(horizon=time_horizon)

        # changing variables to be same as symbolic notation

        if type(self.x0) == np.ndarray:
            x0 = self.x0.flatten().tolist() 
        else:
            x0 = self.x0
        T = self.T
        theta = param_value

        ''' Setting up Direct Collocation '''
        # using casadi's example for direct collocation to solve

        ### Defining Collocation stuff ###
        # degree of interpolating polynomial
        d = 3
        # collocation points
        tau_root = np.append(0, ca.collocation_points(d, 'legendre'))
        # coefficients of collocation eq
        C = np.zeros((d+1, d+1))
        # coefficients of continuity equation
        D = np.zeros(d+1)
        # coefficients of quadrature function
        B = np.zeros(d+1)

        # Construct polynomial basis
        for j in range(d+1):
            # Construct Lagrange polynomials to get the polynomial basis at the collocation point
            p = np.poly1d([1])
            for r in range(d+1):
                if r != j:
                    p *= np.poly1d([1, -tau_root[r]]) / (tau_root[j]-tau_root[r])

            # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
            D[j] = p(1.0)

            # Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation
            pder = np.polyder(p)
            for r in range(d+1):
                C[j,r] = pder(tau_root[r])

            # Evaluate the integral of the polynomial to get the coefficients of the quadrature function
            pint = np.polyint(p)
            B[j] = pint(1.0)

        N = self.N # number of control intervals 
        dt = T/N


        ''' Creating NLP '''
        # Start with an empty NLP
        w = [] # decision variable of NLP
        w0 = []
        lbw = []
        ubw = []
        J = 0
        g = [] # other constraint func ( g_lb <= g(x,p) <= g_ub)
        lbg = []
        ubg = []

        # for seperating x and u trajectories from solution
        x_plot = []
        u_plot = []

        # 'Lift initial conditions
        Xk = ca.MX.sym('X0', self.n_state)
        w.append(Xk)
        lbw.append(x0)
        ubw.append(x0)
        w0.append(x0)
        x_plot.append(Xk)

        # formulate the NLP
        for k in range(N):
            # New NLP variable for the input
            Uk = ca.MX.sym('U_' + str(k), self.n_input)
            w.append(Uk)
            lbw.append(self.input_lb)
            ubw.append(self.input_ub)
            w0.append([0.5*(x + y) for x, y in zip(self.input_lb, self.input_ub)])
            u_plot.append(Uk)

            # State at collocation points
            Xc = []
            for j in range(d):
                Xkj = ca.MX.sym('X_'+str(k)+'_'+str(j), self.n_state)
                Xc.append(Xkj)
                w.append(Xkj)
                lbw.append(self.state_lb)
                ubw.append(self.state_ub)
                w0.append([0.5*(x + y) for x, y in zip(self.state_lb, self.state_ub)]) # need to check what to do here
            
            # Loop over collocation points
            Xk_end = D[0]*Xk
            for j in range(1,d+1):
                # expr for state derivative at collocation point
                xp = C[0,j]*Xk
                for r in range(d): xp = xp + C[r+1,j]*Xc[r]

                # append collocation equations
                fj = self.f_func(Xc[j-1], Uk, theta)
                qj = self.l_func(Xc[j-1], Uk, theta)
                g.append(dt*fj - xp)
                lbg.append([0]*self.n_state)
                ubg.append([0]*self.n_state)

                # add contribution to the end state
                Xk_end = Xk_end + D[j]*Xc[j-1]

                # add contribution to quadrature function
                J = J + B[j]*qj*dt
            
            # New NLP variable for state at end of interval
            Xk = ca.MX.sym('X_' + str(k+1), self.n_state)
            w.append(Xk)
            lbw.append(self.state_lb)
            ubw.append(self.state_ub)
            w0.append([0.5*(x + y) for x, y in zip(self.state_lb, self.state_ub)]) # need to check what to do here
            x_plot.append(Xk)

            # add equality constraint
            g.append(Xk_end - Xk)
            lbg.append([0]*self.n_state)
            ubg.append([0]*self.n_state)

        # add final cost
        J = J + self.h_func(Xk, theta)

        # concatenate vectors
        w = ca.vertcat(*w)
        g = ca.vertcat(*g)
        w0 = np.concatenate(w0)
        lbw = np.concatenate(lbw)
        ubw = np.concatenate(ubw)
        lbg = np.concatenate(lbg)
        ubg = np.concatenate(ubg)

        x_plot = ca.horzcat(*x_plot)
        u_plot = ca.horzcat(*u_plot)

        # create NLP solver
        prob = {'f': J, 'x': w, 'g': g}
        opts = {'ipopt.print_level': print_level, 'ipopt.sb': 'yes', 'print_time': print_level}
        solver = ca.nlpsol('solver', 'ipopt', prob, opts)

        # Function to get x and u trajectories from w
        trajectories = ca.Function('trajectories', [w], [x_plot, u_plot], ['w'], ['x', 'u'])

        # Solve the NLP
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        x_opt, u_opt = trajectories(sol['x'])
        x_opt = x_opt.full() # to numpy array
        u_opt = u_opt.full() # to numpy array

        time = np.linspace(0, T, N+1)

        x = CubicSpline(time, x_opt.T, extrapolate=True)
        u = CubicSpline(time[:-1], u_opt.T, extrapolate=True)

        # output
        opt_sol = {
            'x_traj_opt': x,
            'u_traj_opt': u,
            'theta_value': theta,
            'time': time,
            'horizon': T,
            'cost': sol['f'].full()
        }

        return opt_sol
    
    def ocSolver(self, param_value=1, initial_state=None, time_horizon=1, print_level=0, method=0):
        # Method is Direct Multiple Shooting
        
        assert hasattr(self, 'x'), "Define the state variable first!"
        assert hasattr(self, 'u'), "Define the control variable first!"
        assert hasattr(self, 'f'), "Define the system dynamics first!"
        assert hasattr(self, 'l'), "Define the state cost first!"
        assert hasattr(self, 'h'), "Define the terminal cost first!"

        if not hasattr(self, 'x0'):
            self.setInitialCondition(initial_state)
        if not hasattr(self, 'T'):
            self.setHorizon(horizon=time_horizon)

        # changing variables to be same as symbolic notation

        if type(self.x0) == np.ndarray:
            x0 = self.x0.flatten().tolist()
        else:
            x0 = self.x0
        T = self.T
        theta = param_value

        ''' Setting up Direct Multiple Shooting '''
        # discrete time dynamics with Fixed step Runge-Kutta 4 integrator
        N = self.N # number of control intervals 

        X0 = ca.MX.sym('X0', self.n_state)
        U = ca.MX.sym('U', self.n_input)
        
        if method==0:
            # discrete time dynamics with Fixed step Runge-Kutta 4 integrator
            M = 4 # RK4 steps per interval
            DT = T/N/M
            X = X0
            L = 0
            for j in range(M):
                k1 = self.f_func(X, U, theta)
                k1_l = self.l_func(X, U, theta)
                k2 = self.f_func(X + DT/2 * k1, U, theta)
                k2_l = self.l_func(X + DT/2 * k1, U, theta)
                k3 = self.f_func(X + DT/2 * k2, U, theta)
                k3_l = self.l_func(X + DT/2 * k2, U, theta)
                k4 = self.f_func(X + DT * k3, U, theta)
                k4_l = self.l_func(X + DT * k3, U, theta)

                X = X + DT/6*(k1 + 2*k2 + 2*k3 + k4)
                L = L + DT/6*(k1_l + 2*k2_l + 2*k3_l + k4_l)
            F = ca.Function('F', [X0, U], [X,L], ['x0', 'p'], ['xf', 'qf'])
        
        elif method==1:
            # discretise with a "more advanced integrator" (CVODES)
            X = X0
            f = self.f_func(X,U,theta)
            l = self.l_func(X,U,theta)
            
            dae = {'x': X0, 'p': U, 'ode': f, 'quad': l}
            F = ca.integrator('F', 'cvodes', dae, 0, T/N)
        
        elif method==2:
            # euler discretisation
            DT = T/N
            X = X0
            L = 0

            X = X + DT*(self.f_func(X, U, theta))
            L = L + self.l_func(X, U, theta)
            F = ca.Function('F', [X0, U], [X,L], ['x0', 'p'], ['xf', 'qf'])

        ''' Creating NLP '''
        # Start with an empty NLP
        w = [] # decision variable of NLP
        w0 = []
        lbw = []
        ubw = []
        J = 0
        g = [] # other constraint func ( g_lb <= g(x,p) <= g_ub)
        lbg = []
        ubg = []

        # Lift initial conditions
        Xk = ca.MX.sym('X0', self.n_state)
        w.append(Xk)
        lbw.append(x0)
        ubw.append(x0)
        w0.append(x0)
        # g.append(x0 - Xk)
        # lbg.append([0]*self.n_state)
        # ubg.append([0]*self.n_state)

        # formulate the NLP
        for k in range(N):
            # New NLP variable for the input
            Uk = ca.MX.sym('U_' + str(k), self.n_input)
            w.append(Uk)
            lbw.append(self.input_lb)
            ubw.append(self.input_ub)
            w0.append([0.5*(x + y) for x, y in zip(self.input_lb, self.input_ub)])

            # Integrate til end of interval
            Fk = F(x0=Xk, p=Uk)
            Xk_end = Fk['xf']
            J = J + Fk['qf']
            
            # New NLP variable for state at end of interval
            Xk = ca.MX.sym('X_' + str(k+1), self.n_state)
            w.append(Xk)
            lbw.append(self.state_lb)
            ubw.append(self.state_ub)
            w0.append([0.5*(x + y) for x, y in zip(self.state_lb, self.state_ub)]) # need to check what to do here

            # add equality constraint
            g.append(Xk_end - Xk)
            lbg.append([0]*self.n_state)
            ubg.append([0]*self.n_state)

        # add final cost
        J = J + self.h_func(Xk, theta)

        # concatenate vectors
        w = ca.vertcat(*w)
        g = ca.vertcat(*g)
        w0 = np.concatenate(w0)
        lbw = np.concatenate(lbw)
        ubw = np.concatenate(ubw)
        lbg = np.concatenate(lbg)
        ubg = np.concatenate(ubg)

        # create NLP solver
        prob = {'f': J, 'x': w, 'g': g}
        opts = {'ipopt.print_level': print_level, 'ipopt.sb': 'yes', 'print_time': print_level}
        solver = ca.nlpsol('solver', 'ipopt', prob, opts)

        # Solve the NLP
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

        sol_traj = np.concatenate((sol['x'].full().flatten(), self.n_input * [0]))
        sol_traj = np.reshape(sol_traj, (-1, self.n_state + self.n_input))
        x_opt = sol_traj[:, 0:self.n_state]
        u_opt = np.delete(sol_traj[:, self.n_state:], -1, 0)

        time = np.linspace(0, T, N+1)

        x = CubicSpline(time, x_opt, extrapolate=True)
        u = CubicSpline(time[:-1], u_opt, extrapolate=True)

        # output
        opt_sol = {
            'x_traj_opt': x,
            'u_traj_opt': u,
            # 'lambda_traj_opt': lam,
            'theta_value': theta,
            'time': time,
            'horizon': T,
            'cost': sol['f'].full()
        }

        return opt_sol
    

class Cont_COCSys:
    """
        - define a Continuous Constrained Optimal Control System
        - state and input variables
        - dynamics (ODE)
        - constraints 
        - terminal and stage cost functions
        - parameters (unknown)
    - solve the OC system to obtain the state and input trajectory

    """    
    def __init__(self):
        # self.inf = ca.inf
        self.inf = 1e20

    def setStateVariable(self, state, lower_bound=[], upper_bound=[]):
        self.x = state # match notation
        self.n_state = self.x.numel() # this is 'n'

        if len(lower_bound) == self.n_state:
            self.state_lb = lower_bound
        else:
            self.state_lb = self.n_state * [-self.inf]

        if len(upper_bound) == self.n_state:
            self.state_ub = upper_bound
        else:
            self.state_ub = self.n_state * [self.inf]
    
    def setInputVariable(self, input, lower_bound=[], upper_bound=[]):
        self.u = input # match notation
        self.n_input = self.u.numel() # this is 'm'

        if len(lower_bound) == self.n_input:
            self.input_lb = lower_bound
        else:
            self.input_lb = self.n_input * [-self.inf]

        if len(upper_bound) == self.n_input:
            self.input_ub = upper_bound
        else:
            self.input_ub = self.n_input * [self.inf]

    def setParamVariable(self, param=None):
        # create useless variable if no unknown parameters
        if param is None or param.numel() == 0:
            self.theta = ca.SX.sym('theta')
        else:
            self.theta = param
        self.n_param = self.theta.numel()
    
    def setDynamics(self, ODE):
        # want dynamics as an ODE
        # dx/dt = f(x,u) 

        if not hasattr(self, 'theta'):
            self.setParamVariable()
        
        # keep symbolic expression (of LHS dynamics)
        self.f = ODE
        # create CasADI function object
        self.f_func = ca.Function('dynamics', [self.x, self.u, self.theta], [self.f])

    def setMeasurement(self, measurement_func=None):
        # y = g(x,u)
        if not hasattr(self, 'theta'):
            self.setParamVariable()
        
        if measurement_func is None:
            self.g = ca.vertcat(self.x, self.u)
        else:
            self.g = measurement_func
        
        self.g_func = ca.Function('measurements', [self.x, self.u], [self.g])
    
    def setStageCost(self, stage_cost):
        if not hasattr(self, 'theta'):
            self.setParamVariable()
        
        assert stage_cost.numel() == 1, "stage_cost must be a scalar function"

        # keep symbolic expression
        self.l = stage_cost
        # create CasADI function object
        self.l_func = ca.Function('stage_cost', [self.x, self.u, self.theta], [self.l])
    
    def setTerminalCost(self, term_cost):
        if not hasattr(self, 'theta'):
            self.setParamVariable()
        
        assert term_cost.numel() == 1, "term_cost must be a scalar function"

        # keep symbolic expression
        self.h = term_cost
        # create CasADI function object
        self.h_func = ca.Function('terminal_cost', [self.x, self.theta], [self.h])

    def setStageInequCstr(self, stage_inequality=None):
        if not hasattr(self, 'theta'):
            self.setParamVariable()

        self.stage_inequ = stage_inequality

        if self.stage_inequ is not None:
            self.stage_inequ_func = ca.Function(
                'stage_inequality_constraint',
                [self.x, self.u, self.theta],
                [self.stage_inequ]
            )
            self.n_stage_inequ = self.stage_inequ_func.numel_out()
        else:
            self.n_stage_inequ = 0

    def setStageEquCstr(self, stage_equality=None):
        if not hasattr(self, 'theta'):
            self.setParamVariable()

        self.stage_equ = stage_equality

        if self.stage_equ is not None:
            self.stage_equ_func = ca.Function(
                'stage_equality_constraint',
                [self.x, self.u, self.theta],
                [self.stage_equ]
            )
            self.n_stage_equ = self.stage_equ_func.numel_out()
        else:
            self.n_stage_equ = 0

    # set the final inequality constraints (if exists)
    def setTerminalInequCstr(self, terminal_inequality=None):
        if not hasattr(self, 'theta'):
            self.setParamVariable()

        self.term_inequ = terminal_inequality

        if self.term_inequ is not None:
            self.term_inequ_func = ca.Function(
                'terminal_inequality_constraint',
                [self.x, self.u, self.theta],
                [self.term_inequ]
            )
            self.n_term_inequ = self.term_inequ_func.numel_out()
        else:
            self.n_term_inequ = 0

    # set the final equality constraints (if exists)
    def setTerminalEquCstr(self, terminal_equality=None):
        if not hasattr(self, 'theta'):
            self.setParamVariable()

        self.term_equ = terminal_equality

        if self.term_equ is not None:
            self.term_equ_func = ca.Function(
                'terminal_equality_constraint',
                [self.x, self.u, self.theta],
                [self.term_equ]
            )
            self.n_term_equ = self.term_equ_func.numel_out()
        else:
            self.n_term_equ = 0
    
    def setInitialCondition(self, initial_state=None):
        self.x0 = initial_state

    def setHorizon(self, horizon=1, num_traj_points=20):
        self.T = horizon
        self.N = num_traj_points
    
    def ocSolver(self, param_value=1, initial_state=None, time_horizon=1, print_level=0, method=0):
        assert hasattr(self, 'x'), "Define the state variable first!"
        assert hasattr(self, 'u'), "Define the control variable first!"
        assert hasattr(self, 'f'), "Define the system dynamics first!"
        assert hasattr(self, 'l'), "Define the state cost first!"
        assert hasattr(self, 'h'), "Define the terminal cost first!"

        if not hasattr(self, 'term_equ'):
            self.setTerminalEquCstr()
        if not hasattr(self, 'term_inequ'):
            self.setTerminalInequCstr()
        if not hasattr(self, 'stage_equ'):
            self.setStageEquCstr()
        if not hasattr(self, 'stage_inequ'):
            self.setStageInequCstr()

        if not hasattr(self, 'x0'):
            self.setInitialCondition(initial_state)
        if not hasattr(self, 'T'):
            self.setHorizon(horizon=time_horizon)

        # changing variables to be same as symbolic notation

        if type(self.x0) == np.ndarray:
            x0 = self.x0.flatten().tolist() 
        else:
            x0 = self.x0
        T = self.T
        theta = param_value

        ''' Setting up Direct Multiple Shooting '''
        # discrete time dynamics with Fixed step Runge-Kutta 4 integrator
        N = self.N # number of control intervals 

        X0 = ca.MX.sym('X0', self.n_state)
        U = ca.MX.sym('U', self.n_input)
        
        if method==0:
            # discrete time dynamics with Fixed step Runge-Kutta 4 integrator
            M = 4 # RK4 steps per interval
            DT = T/N/M
            X = X0
            L = 0
            for j in range(M):
                k1 = self.f_func(X, U, theta)
                k1_l = self.l_func(X, U, theta)
                k2 = self.f_func(X + DT/2 * k1, U, theta)
                k2_l = self.l_func(X + DT/2 * k1, U, theta)
                k3 = self.f_func(X + DT/2 * k2, U, theta)
                k3_l = self.l_func(X + DT/2 * k2, U, theta)
                k4 = self.f_func(X + DT * k3, U, theta)
                k4_l = self.l_func(X + DT * k3, U, theta)

                X = X + DT/6*(k1 + 2*k2 + 2*k3 + k4)
                L = L + DT/6*(k1_l + 2*k2_l + 2*k3_l + k4_l)
            F = ca.Function('F', [X0, U], [X,L], ['x0', 'p'], ['xf', 'qf'])
        
        elif method==1:
            # discretise with a "more advanced integrator" (CVODES)
            X = X0
            f = self.f_func(X,U,theta)
            l = self.l_func(X,U,theta)
            
            dae = {'x': X0, 'p': U, 'ode': f, 'quad': l}
            F = ca.integrator('F', 'cvodes', dae, 0, T/N)
        
        elif method==2:
            # euler discretisation
            DT = T/N
            X = X0
            L = 0

            X = X + DT*(self.f_func(X, U, theta))
            L = L + self.l_func(X, U, theta)
            F = ca.Function('F', [X0, U], [X,L], ['x0', 'p'], ['xf', 'qf'])

        ''' Creating NLP '''
        # Start with an empty NLP
        w = [] # decision variable of NLP
        w0 = []
        lbw = []
        ubw = []
        J = 0
        g = [] # other constraint func ( g_lb <= g(x,p) <= g_ub)
        lbg = []
        ubg = []

        # 'Lift' initial conditions
        Xk = ca.MX.sym('X0', self.n_state)
        w.append(Xk)
        lbw.append(x0)
        ubw.append(x0)
        w0.append(x0)


        # formulate the NLP
        for k in range(N):
            # New NLP variable for the input
            Uk = ca.MX.sym('U_' + str(k), self.n_input)
            w.append(Uk)
            lbw.append(self.input_lb)
            ubw.append(self.input_ub)
            w0.append([0.5*(x + y) for x, y in zip(self.input_lb, self.input_ub)])

            # Enforce stage inequality constraint (if it exists)
            if self.stage_inequ is not None:
                g.append(self.stage_inequ_func(Xk, Uk, theta))
                lbg.append([-self.inf]*self.n_stage_inequ)
                ubg.append([0]*self.n_stage_inequ)

            # Enforce stage equality constraint (if it exists)
            if self.stage_equ is not None:
                g.append(self.stage_equ_func(Xk, Uk, theta))
                lbg.append([0]*self.n_stage_equ)
                ubg.append([0]*self.n_stage_equ)

            # Integrate til end of interval
            Fk = F(x0=Xk, p=Uk)
            Xk_end = Fk['xf']
            J = J + Fk['qf']
            
            # New NLP variable for state at end of interval
            Xk = ca.MX.sym('X_' + str(k+1), self.n_state)
            w.append(Xk)
            lbw.append(self.state_lb)
            ubw.append(self.state_ub)
            w0.append([0.5*(x + y) for x, y in zip(self.state_lb, self.state_ub)]) 

            # add dynamics equality constraint
            g.append(Xk_end - Xk)
            lbg.append([0]*self.n_state)
            ubg.append([0]*self.n_state)
        
        # enforce terminal inequality constraint (if it exists)
        if self.term_inequ is not None:
            g.append(self.term_inequ_func(Xk, Uk, theta))
            lbg.append([-self.inf]*self.n_term_inequ)
            ubg.append([0]*self.n_term_inequ)

        # enforce terminal equality constraint (if it exists)
        if self.term_equ is not None:
            g.append(self.term_equ_func(Xk, Uk, theta))
            lbg.append([0]*self.n_term_equ)
            ubg.append([0]*self.n_term_equ)


        # add terminal cost
        J = J + self.h_func(Xk, theta)

        # concatenate vectors
        w = ca.vertcat(*w)
        g = ca.vertcat(*g)
        w0 = np.concatenate(w0)
        lbw = np.concatenate(lbw)
        ubw = np.concatenate(ubw)
        lbg = np.concatenate(lbg)
        ubg = np.concatenate(ubg)

        # create NLP solver
        prob = {'f': J, 'x': w, 'g': g}
        opts = {'ipopt.print_level': print_level, 'ipopt.sb': 'yes', 'print_time': print_level}
        solver = ca.nlpsol('solver', 'ipopt', prob, opts)

        # Solve the NLP
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

        sol_traj = np.concatenate((sol['x'].full().flatten(), self.n_input * [0]))
        sol_traj = np.reshape(sol_traj, (-1, self.n_state + self.n_input))
        x_opt = sol_traj[:, 0:self.n_state]
        u_opt = np.delete(sol_traj[:, self.n_state:], -1, 0)

        time = np.linspace(0, T, N+1)

        x = CubicSpline(time, x_opt, extrapolate=True)
        u = CubicSpline(time[:-1], u_opt, extrapolate=True)

        # output
        opt_sol = {
            'x_traj_opt': x,
            'u_traj_opt': u,
            # 'lambda_traj_opt': lam,
            'theta_value': theta,
            'time': time,
            'horizon': T,
            'cost': sol['f'].full()
        }

        return opt_sol
    