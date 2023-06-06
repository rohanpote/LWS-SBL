function [xhat,kpset,compute_time]=OMP_QRdecompxhat(y,A,suppsize)
% OMP Code written by Rohan R. Pote (Sept 2022, modified May 2023)
% Sparse vector computed using QR decomposition
% y: measurement vector or matrix
% A: measurement matrix
% lambda_n: noise variance parameter
% suppsize: sparse support set size to be recovered

m=size(A,1);
n=size(A,2);
% Initialization
bp = y;
Qp = zeros(m,suppsize);
kpset = zeros(1,suppsize);
ynorm=zeros(suppsize,1);
Rmat=zeros(suppsize);
xhat = zeros(n,1);

% OMP Support Recovery
tic
for p=1:suppsize
    Atbp = A'*bp;
    Atbpabs = abs(Atbp);
    [~, kp] = max(Atbpabs./(vecnorm(A).')); %kp=candidateset(kp);
    kpset(p) = kp;
    if p==1
        qkp = A(:,kp);
    else
        Rmat(1:p-1,p)=Qp(:,1:p-1)'*A(:,kp);
        qkp = A(:,kp)-Qp(:,1:p-1)*Rmat(1:p-1,p);
    end
    Rmat(p,p)=vecnorm(qkp);
    qkp = qkp/Rmat(p,p);
    Qp(:,p) = qkp;
    ynorm(p)=(qkp'*bp); % used later to solve Normal equations
    bp = bp-qkp*ynorm(p);
    %             toc
end

% Computing Sparse Vector x
opts.UT=true;
xhat(kpset) = linsolve(Rmat,ynorm,opts);
compute_time=toc;
end