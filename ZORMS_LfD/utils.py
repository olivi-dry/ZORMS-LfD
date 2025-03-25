import numpy as np

def project_cone(X, d=0, diag=False):
    if diag:
        X[X<d] = 0
        return X
    else:
        lam, v = np.linalg.eig(X)
        lam[lam<d] = d

        proj = v@np.diag(lam)@v.T

        return proj

def gen_GOE(n, diagonal=False):
    """
    Generate an nxn matrix from the Gaussian Orthogonal Ensemble

    Parameters
    ----------
    n : int
        size of matrix

    Returns
    -------
    ndarray
        A symmetric 
    """

    diag = np.random.standard_normal(n)
    if diagonal:
        GOE = np.diag(diag)
    else:
        A = np.random.normal(0,0.5,(n,n))
        A = np.tril(A, -1)
        # GOE = 0.5*(A+A.T)
        GOE = (A+A.T)
        np.fill_diagonal(GOE,diag)

    return GOE

def gen_special_GOE(n, PSD=False, PD=False, diag=False):
    """
    This can be used to generate a U that is positive or semi-positive definite.
    Repeatedly

    Parameters
    ----------
    n : _type_
        _description_
    SPD : bool, optional
        _description_, by default False
    PD : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_
    """
    condition = False
    iter = 0
    
    while condition != True:
        U = gen_GOE(n, diagonal=diag)
        iter += 1
        lam, v = np.linalg.eig(U)
        if PSD:
            if not np.any(lam<0):
                condition = True
        elif PD:
            if not np.any(lam<=0):
                condition = True
        else:
            condition = True
    
    return U

def params_convex(n, L0, r, e):
    mu = e/(L0*np.sqrt(2*(n**2 + n)))
    alpha = 2*r/(L0*np.sqrt(n**4+2*n**3+5*n**2+4*n))
    N = (n**4+2*n**3+5*n**2+4*n)*L0**2*r**2/(e**2)

    print(f"mu <= {mu:0.3e}")
    print(f"alpha_k = {alpha:0.3e}/sqrt(k+1)")
    print(f"N >= {N:0.1e}")

def params_nonconvex(n, L0, r, e, d):
    mu = e/(L0*np.sqrt(0.5*(n**2 + n)))
    alpha = np.sqrt(8*e*r/(L0**3*(n**2 + n)*(n**4 + 2*n**3 + 5*n**2 + 4*n)))
    N = (n**2 + n)*(n**4 + 2*n**3 + 5*n**2 + 4*n)*L0**5*r/(2*e*d**2)

    print(f"mu <= {mu:0.3e}")
    print(f"alpha = {alpha:0.3e}/sqrt(N+1)")
    print(f"N >= {N:0.1e}")

def params_vector(n, L0, r, e):
    mu = e/(2*L0*np.sqrt(n))
    alpha = r/(L0*(n + 4))
    N = (4*(n + 4)**2)*L0**2*r**2/(e**2)

    print(f"mu <= {mu:0.3e}")
    print(f"alpha_k = {alpha:0.3e}/sqrt(k+1)")
    print(f"N >= {N:0.1e}")
