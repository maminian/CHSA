
%%Create n points in the plane, with additional points along boundary of
%%convex hull, then use CHSA to stratify points based on proximity to
%%boundary, create figure colored such that points with negative weights
%%are cyan and then a figure that colors points according to magnitude of
%%2-norm of associated weight vector

%Create random data and use built-in MATLAB convex hull algorithm to
%visualize data and vertices of convex hull
n=50
X=rand(2,n);
CH=convhull(X(1,:),X(2,:));

%%%Add in some points on the boundaries
NumVer=length(CH);
m=10;%Add in m points on boundary
MorePoints=zeros(2,m);
for i=1:m
    Ind=randi(NumVer); %Choose random vertex of convex hull
    Num=rand(1);
    if Ind<NumVer
        MorePoints(:,i)=Num*X(:,CH(Ind))+(1-Num)*X(:,CH(Ind+1));
    else
        MorePoints(:,i)=Num*X(:,CH(Ind))+(1-Num)*X(:,CH(2));
    end
end

X=[X MorePoints];
[d,p]=size(X);

%Make color for plot
Color=repmat([0 0 0],p,1);
NumNeg=zeros(1,p);
for i=1:p
    if isempty(find(CH==i))~=1
        Color(i,:)=[1 0 0]; %vertex point
    elseif i>n
        Color(i,:)=[1 0 1]; %point on boundary
    end
end
figure
hold on
plot(X(1,CH),X(2,CH),'-r')
scatter(X(1,:),X(2,:),50,Color,'filled')
hold off

%Convex Hull Stratification Algorithm
[Y, WtildeMatrix,Indices]=ConvexHull(X,20,10^-3,10^-6); 

%Color points such that those with weight vector with negative entry are
%cyan
figure
hold on
scatter(X(1,:),X(2,:),50,[0,0,0], 'filled')
scatter(X(1,Indices),X(2,Indices),50, [0,1,1], 'filled')
hold off

%%%Color points according to magnitude of 2-norm
NormVec=zeros(1,p);
for i=1:p
    NormVec(i)=norm(WtildeMatrix(:,i),2);
end

[Val,Ind]=sort(NormVec);
figure
scatter(X(1,Ind),X(2,Ind),50,Val,'filled')
colormap(flipud(autumn));
colorbar


