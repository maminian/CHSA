def PDIPAQuad(Q,c,A,b, tol=10**-4, minstep=10**-8, **kwargs):
    '''
    Implements the primal dual interior point (path-following)
    algorithm for solving the quadratic program of the form

    minimize 0.5 * (x,Qx) + (c,x)
        subject to  Ax=b
                    x>=0


    This is transliterated from Lori Ziegelmeier's Matlab
    implementation PDIPAQuad.m

    Inputs:
        Q : array shape (N,N) of the quadratic term
        c : array shape (N,) of the linear term
        A : array shape (M,N) of M linear (affine) constraints
        b : array shape (M,) of the corresponding right hand sides of constraints

    Optional inputs:
        tol : float. Convergence criterion for the error. (default: 10**-4)
        minstep : float. Breaks loop early if stepsize is below this threshold. (default: 10**-8)

        verbosity : integer indicating level of output (default: 0)
        maxiter : maximum number of iterations (default: 1000)
        output_diagnostics : Whether to output a dictionary
            with detailed convergence information. (Default: False)
    Outputs:
        x : array shape (N,) of decision variables
        output : if output_diagnostics==True, then this second output
            is a dictionary containing detailed convergence information.
    '''
    import numpy as np
    import pdb

    verbosity = kwargs.get('verbosity', 0)
    maxiter = kwargs.get('maxiter', 1000)

    M,N = np.shape(A)

    # initialization
    x = np.ones(N)
    lam = np.ones(N)
    nu = np.ones(M)
    Theta = np.concatenate([x,lam,nu])  # initial point

    # reform (?)
    X = np.diag(x)
    e = np.ones(N)
    Lambda = np.diag(lam)

    # calculate the residuals
    rho = np.dot(A,x) - b   # primal residual
    sigma = np.dot(A.T, nu) - lam + c + np.dot(Q,x) # dual residual
    gamma = np.dot(X,lam)   # complementarity
    mu = gamma/(5*N) # ????????

    # normed residuals
    m1 = np.linalg.norm(rho,1)
    m2 = np.linalg.norm(sigma,1)
    m3 = np.linalg.norm(gamma,1)
    err = max(m1,m2,m3)

    stepsizes = []
    m1s = [m1]
    m2s = [m2]
    m3s = [m3]
    xs = [x]
    lams = [lam]
    nus = [nu]

#    pdb.set_trace()

    for i in range(maxiter):

        if err < tol:
            if verbosity>0:
                print('Solution found to within tolerance in %i iterations.'%i)
            break
        #

        # Create linear system of equations DF DelT = -F
        DF1 = np.hstack([Q, -np.eye(N), A.T])
        DF2 = np.hstack([Lambda, X, np.zeros((N,M))])
        DF3 = np.hstack([A,np.zeros((M,N)),np.zeros((M,M))])

        DF = np.vstack([DF1,DF2,DF3])

        F = np.hstack( [
            np.dot(A.T, nu) - lam + c + np.dot(Q,x),
            np.dot(X,lam) - mu*e,
            np.dot(A,x) - b
        ] )

        # solve for the update
        DelT = np.linalg.solve(DF,-F)

        # Determine the change in each of the variables.
        Delx = DelT[:N]
        Dellam = DelT[N:2*N]
        Delnu = DelT[2*N:2*N+M]

        mm = max([ max(-Delx/x), max(-Dellam/lam) ])  # Neglecting -Delnu/nu

        stepsize = min(0.9/mm, 1)
        stepsizes.append( stepsize )

        if stepsize < minstep:
            if verbosity>0:
                print('Step size %.3e is below the threshold %.3e; stopping.'%(stepsize,minstep))
            break
        #

        # Update solution
        ThetaNew = Theta + stepsize*DelT    # Update primal
        x = ThetaNew[:N]
        lam = ThetaNew[N:2*N]
        nu = ThetaNew[2*N:2*N+M]
        Theta = ThetaNew
        X = np.diag(x)
        Lambda = np.diag(lam)

        # Calculate the residuals
        rho = np.dot(A,x) - b
        sigma = np.dot(A.T,nu) - lam + c + np.dot(Q,x)
        gamma = np.dot(x,lam)

        mu = gamma/(5*N)

        m1s.append( np.linalg.norm(rho,1) )
        m2s.append( np.linalg.norm(sigma,1) )
        # m3s.append( np.linalg.norm(gamma,1) )
        m3s.append( np.abs(gamma) ) # watch this with more constraints
        err = max(m1s[-1], m2s[-1], m3s[-1])

        xs.append(x)
        nus.append(nu)
        lams.append(lam)
    #

    if (i==maxiter-1) and (err>=tol) and (verbosity>0):
        print('The maximum number of iterations has been reached')
    #

    # Create dictionary with convergence information
    if kwargs.get('output_diagnostics', False):
        quadv = np.dot(x, np.dot(Q,x))

        output = {}
        output['niter'] = i
        output['stepsizes'] = np.array(stepsizes)
        output['complementarity'] = np.array(m1s)
        output['dual feasibility'] = np.array(m2s)
        output['primal feasibility'] = np.array(m3s)
        output['objective'] = 0.5*quadv + np.dot(c,x)
        output['bounded'] = np.dot(c,x) + np.dot(nu,b) + quadv  # duality gap
        output['x history'] = np.array(xs)
        output['nu history'] = np.array(nus)
        output['lambda history'] = np.array(lams)

        return x,output
    else:
        return x
    #
#

if __name__=="__main__":
    # quick and dirty test; two dimensions,
    # one affine constraint:
    #
    # min 0.5*(x**2 + y**2)
    # s.t. -x - 2y <= 1
    #
    # solution should lie on the boundary
    # of the constraint; -x-2y = 1, where
    # the Lagrange multiplier constraint should
    # give necessary condition that the solution
    # is parallel to (1,2).

    import numpy as np
    from matplotlib import pyplot

    Q = np.eye(2)
    Q[0,0] = 0.5
    c = np.zeros(2)
    A = np.array([[1,1]])   # er... signs?
    b = np.array([8])

    x = PDIPAQuad(Q,c,A,b)

    def qf(xv):
        return 0.5*np.dot(xv, np.dot(Q,xv))
    #
    def constr(xv, tol=0.08):
        # returns true or false depending on whether the constraints are satisfied
        out = all(
                (np.abs(np.dot(A,xv) - b) < tol).flatten()
            )
        return out
    #

    fig,ax = pyplot.subplots(1,1)

    xg,yg = np.meshgrid( np.linspace(-1,7,101), np.linspace(-1,7,101) )

    zg = np.array( [[ qf(np.array([xg[i,j],yg[i,j]])) for j in range(xg.shape[1])]  for i in range(xg.shape[0])] )
    ineq = np.array( [[ constr(np.array([xg[i,j],yg[i,j]])) for j in range(xg.shape[1])]  for i in range(xg.shape[0])] )

    mycm = pyplot.cm.Reds
    mycm.set_over(color=[1,0,0,1])
    mycm.set_under(color=[0,0,0,0])
    ax.contourf(xg,yg,zg, 21)
    ax.contourf(xg,yg,ineq, cmap=mycm, vmin=0.1, vmax=0.9)
    ax.axis('square')

    # Look at the computed solution; does it lie on the
    # constraint, tangent to the level curves of the quadratic form?
    ax.scatter(x[0],x[1], c=[1,1,1], s=100, marker='s')

    fig.show()
    pyplot.ion()

#
