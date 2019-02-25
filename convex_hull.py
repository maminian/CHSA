def convex_hull(X,K,lam,gamma,**kwargs):
    '''
    A transliteration of Lor Ziegelmeier's ConvexHull.m code.
    Primary difference is that **the data matrix X should be
    arranged by rows**, as is typical in python/sklearn/calcom.

    The convex hull stratification algorithm detailed in
    the article

        L. Ziegelmeier, M. Kirby, C. Peterson,
        "Stratifying High-Dimensional Data Based on proximity
        to the Convex Hull Boundary", SIAM Review, Vol. 59, No. 2, pp.346-365, 2017

    Inputs:
        X : p-by-D dataset X
        K : number of nearest neighbors
        lam : parameter on the l_1 convexity term
        gamma : parameter on the l_2 uniformity term

    Outputs:
        Y : candidates for vertices of the convex hull
        WtildeMatrix : the K-by-p matrix corresponding to weight
            vectors associated to each point (note weight vectors
            with negative entries are candidate vertices)
        Indices : indices of the original data set X with negative
            entries in the weight vector

    Optional inputs:
        verbosity : integer. Set greater than zero for print statements
            (default: 0)
        metric : string, indicating the metric to be used. This is
            passed on to sklearn.metrics.pairwise_distances();
            see that documentation for the full list of options.
            Notable options include 'cosine', 'l1', and 'l2'.
            (default: 'euclidean')
    '''
    import numpy as np
    from sklearn import metrics

    import opt  # contains quadratic program optimization routine

    verbosity = kwargs.get('verbosity',0)
    metric = kwargs.get('metric', 'euclidean')

    p,D = np.shape(X)
    if verbosity>0: print('Computing distance matrix')

    Distance = metrics.pairwise_distances(X, metric=metric)
    # We want a sorted ilst of nearest neighbors for each
    # point.
    nneighbors = [np.argsort(row) for row in Distance]

    if verbosity>0: print('Done computing distance matrix')

    NeighborInd = np.zeros((p,K), dtype=int)
    for j in range(p):
        DiffPoints = np.where( Distance[j]>0. )[0]
        idx = 0
        for k in nneighbors[j]:
            if Distance[j][k]>0:
                NeighborInd[j][idx] = k
                idx += 1
            if idx==K:
                break
        #
    #
    if verbosity>0: print('Done computing nearest neighbors')

    # Creating data cube containing the matrix of
    # neighborhoods for each point X_i
    #
    N = np.zeros( (p,K,D) )
    for i in range(p):
        N[i] = X[NeighborInd[i]]
    #

    # Computing the reconstruction weights by solving
    # a quadratic program.
    # Decision variables are the nonzero entries of W,
    # wtilde, rewritten as wtildeplus and wtildeminus
    # Interested in the negative entries, as should indicate
    # points on boundary or vertices.
    if verbosity>0: print('Forming the quadratic program')
    H = np.zeros((p, 2*K, 2*K))
    f = np.zeros((p, 2*K))

    # to be used with the Kronecker product
    tiling_array = np.array([[1,-1],[-1,1]])

    for i in range(p):
        # Not the best naming scheme, but whatever
        doti = np.dot(X[i],N[i].T)      # dot products of point of interest onto its neighbors
        NNgram = np.dot(N[i], N[i].T)   # dot products between all neighbors (Gram matrix)

        ftilde = -2*doti + lam
        fhat = 2*doti + lam

        Hhat = NNgram
        Htilde = np.kron(tiling_array, Hhat)

        H[i] = Htilde
        f[i] = np.concatenate( [ftilde,fhat] )
    #

    if verbosity>0: print('Done forming quadratic program')
    if verbosity>0: print('Solving quadratic program')


    Aeq = np.kron([1,-1], np.ones(K))
    Aeq.shape = (1,len(Aeq))
    b=np.array([1]);
    WtildeMatrix = np.zeros((p,K))
    fval=0;
    Y=[];
    Indices=[];

#    import pdb
#    pdb.set_trace()

    for i in range(p):
        Htilde = H[i]
        ftilde = f[i]

        Q = 2*(Htilde + gamma*np.kron(tiling_array,np.eye(K)))
        x = opt.PDIPAQuad(Q, ftilde, Aeq, b, 10**-8)

        # Reconstruct weights
        Wtilde = x[:K] - x[K:2*K]
        Wtilde /= sum(Wtilde)
        WtildeMatrix[i] = Wtilde

        # Determining if the weight vector has negative entries
        loc = np.where(Wtilde<0.)[0]

        if len(loc)==0:
            Y.append(X[i])
            Indices.append(i)
        else:
            # print(loc,i)
            pass
        #

        # Checking if each point recnostructed perfectly
        # Verify = np.linalg.norm(X[i] - N[i])**2
    #

    if verbosity>0: print('Done solving the quadratic program')

    return Y,WtildeMatrix,Indices
#

if __name__=="__main__":
    import numpy as np
    from matplotlib import pyplot

    th = np.linspace(0,2*np.pi,51)
    x = np.cos(th) + 0.2*np.random.randn(51)
    y = np.sin(th) + 0.2*np.random.randn(51)
    X = np.array([[xv,yv] for xv,yv in zip(x,y)])

    Y,Wt,Indices = convex_hull(X,4,1,1)

    fig,ax = pyplot.subplots(1,1)
    ax.scatter(x,y)
    ax.axis('square')
    fig.show()
    pyplot.ion()
#
