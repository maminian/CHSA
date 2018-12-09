function [Y, WtildeMatrix,Indices]=ConvexHull(X,K,lambda,gamma)

%Convex Hull stratification algorithm detailed in article
%L. Ziegelmeier, M. Kirby, C. Peterson, "Stratifying High-Dimensional Data 
%Based on Proximity to the Convex Hull Boundary," SIAM Review, Vol. 59, No.
%2, pp. 346-365, 2017.

%Code by Lori Ziegelmeier, note this code has not been optimized

%Input:
% X = Dxp data set X
% K = the number of nearest neighbors
% lambda = parameter on the l_1 convexity term 
% gamma = parameter on the l_2 uniformity term

%Outputs: 
% Y = candidates for vertices of the convex hull
% WtildeMatrix = the Kxp matrix corresponding to weight vectors associated 
%                to each point (note weight vectors with negative entries 
%                are candidate vertices)
% Indices = indices of the original data set X with negative entries in the
%           weight vector


%warning('off','all');


S=size(X);
p = S(2) %Number of points
D= S(1) %Dimensionality


%%%Computing the nearest neighbors
tic;
disp('Computing distance matrix')
SumSqX=sum(X.^2,1);  
SqDist=repmat(SumSqX,p,1)+repmat(SumSqX',1,p)-2*X'*X; %Computes the distance squared between all points using the equation ||x-y|| = sqrt(||x||+||y||-2< x,y>)
Distance =sqrt(SqDist); %finding the distance between all of the points
[SortDist, Index] = sort(Distance); %Sorts each row of the distance matrix
disp('Done computing distance matrix')
%Must use when there are repeated points
NeighborInd=zeros(K,p);
for j=1:p
    DiffPoints=find(SortDist(:,j)); %finds all nonzero distances
    NeighborInd(:,j)=Index(DiffPoints(1):DiffPoints(1)+K-1,j);
end
NeighborInd;
disp('Done Computing Nearest Neighbors')
toc


%Creating a data cube containing the matrix of neighborhoods for each
%point Xi
N=zeros(D,K,p);
for i=1:p
    Ni=zeros(D,K);
    for t=1:K
        Ni(:,t)=X(:,NeighborInd(t,i));
    end
    N(:,:,i)=Ni;
end
N;

%Computing the Reconstruction Weights by solving a quadratic program
%Decision variables are the nonzero entries of W, wtilde, rewritten as
%wtildeplus and wtildeminus
%Interested in the negative entries, as should indicate points on boundary
%or vertices

%Forming the H matrix and the f vector
disp('Forming the Quadratic Program')
tic
H=zeros(2*K,2*K,p);
f=zeros(2*K,p);
for i=1:p
    Ni=N(:,:,i); %Forming a matrix of all neighbors of the point Xi
    Hhat=zeros(K,K);
    ftilde=zeros(K,1);
    fhat=zeros(K,1);
    Xi=X(:,i);
    for j=1:K
        Xj=Ni(:,j);
        ftilde(j)=-2*Xi'*Xj+lambda;
        fhat(j)=2*Xi'*Xj+lambda;
        for l=1:K
            Xl=Ni(:,l);
            Hhat(j,l)=Xj'*Xl;
        end
    end
    Htilde=[Hhat -Hhat; -Hhat Hhat];
    H(:,:,i)=Htilde;
    f(:,i)=[ftilde; fhat];
end
toc
disp('Done forming quadratic program')

disp('Solving the Quadratic Program')
tic
Aeq=[ones(1,K) -ones(1,K)];
b=1;
WtildeMatrix=zeros(K,p);
fval=0;
Y=[];
Indices=[];
for i=1:p
    i;
    Htilde=H(:,:,i);
    ftilde=f(:,i);
    %[x,fval,exitflag,output,lambda] = quadprog(2*(Htilde+gamma*[eye(K) -eye(K); -eye(K) eye(K)]),ftilde,-eye(2*K),zeros(2*K,1),Aeq,b,'options','interior-point-convex');
    
    [output, x,lambdaout, nu] = PDIPAQuad(2*(Htilde+gamma*[eye(K) -eye(K); -eye(K) eye(K)]),ftilde,Aeq,b,10^-10);
    
    Wtildeplus=x(1:K);
    Wtildeminus=x(K+1:2*K);
    Wtilde=Wtildeplus-Wtildeminus;
    Wtilde=Wtilde/sum(Wtilde);
    WtildeMatrix(:,i)=Wtilde;
    if isempty(find(Wtilde<0))==0 %determining if the weight vector has negative entries
        Y=[Y X(:,i)];
        Indices=[Indices i];
    end
    
    Verify=norm(X(:,i)-N(:,:,i)*Wtilde)^2; %Make sure that each point reconstructed perfectly
   
end
toc
disp('Done solving the Quadratic Program')