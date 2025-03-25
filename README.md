# ZORMS-LfD
TODO

## Overview
The codebase includes the following folders:
- **ZORMS_LfD**: a package for the implementation of the *ZORMS-LfD* algorithm:
    - `ZORMS_LfD.py` implements the ZORMS algorithm for learning tasks. Includes the classes
        - `ZORMS_disc` integrates with `PDP.OC_sys` or `SafePDP.COC_sys` classes (vector parameters)
        - `ZORMS_cont` integrates with `COC_Sys.Cont_OCSys` or `COC_Sys.Cont_COCSys` classes (vector parameters)
    - `COC_Sys.py` implements solvers for a continuous-time optimal control system using [CasADi](https://web.casadi.org/)
- **Examples**: a folder containing different examples for the implementation of *ZORMS-LfD*


## Ownership Details
<!-- We have included code owned by other parties to facilitate comparison between our method and previous (gradient-based) methods. -->
We have included code from other codebases to facilitate comparison between our method and previous (gradient-based) methods.
Specifically:

From the *Pontryagin-Differentiable-Programming* [codebase](https://github.com/wanxinjin/Pontryagin-Differentiable-Programming):
- `PDP.py`

From the *Safe-PDP* [codebase](https://github.com/wanxinjin/Safe-PDP):
- `SafePDP.py`
- `JinEnv.py` 

From the *Learning-from-Sparse-Demonstrations* [codebase](https://github.com/wanxinjin/Learning-from-Sparse-Demonstrations):
- `CPDP.py`

From the *Implicit-Diff-Optimal-Control* [codebase](https://github.com/mingu6/Implicit-Diff-Optimal-Control):
- `IDOC_eq.py`
- `IDOC_ineq.py`

The other files in the **PDP** and **IDOC** folders were created by us and implement the respective methods for inverse optimal control.



