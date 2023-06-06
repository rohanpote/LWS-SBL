function [xhat,kpset,compute_time]=OMP_pseudoinvxhat(y,A,suppsize)
% OMP Code written by Rohan R. Pote (Sept 2022, modified May 2023)
% Sparse vector computed using MATLAB pseudo-inverse command
% y: measurement vector or matrix
% A: measurement matrix
% lambda_n: noise variance parameter
% suppsize: sparse support set size to be recovered


m=size(A,1);
n=size(A,2);
% Initialization
bp = y;
qkp = zeros(m,0);
Qp = qkp;
kpset = zeros(1,suppsize);
xhat = zeros(n,1);

% OMP Support Recovery
tic
for p=1:suppsize
    Atbp = A'*bp;
    Atbpabs = abs(Atbp);
    [~, kp] = max(Atbpabs./(vecnorm(A).'));
    kpset(p) = kp;
    qkp = A(:,kp)-Qp*(Qp'*A(:,kp));
    qkp = qkp/vecnorm(qkp);
    Qp = [Qp qkp];
    bp = bp-qkp*(qkp'*bp);
end
xhat(kpset) = pinv(A(:,kpset))*y;
compute_time=toc;
end