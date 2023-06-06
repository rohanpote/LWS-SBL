function [xhat,kpset,gamma_est,compute_time]=LightWeightSeqSBL(y,A,lambda_n,suppsize)
% Proposed Light-Weight Seqential SBL code. Code written by Rohan R. Pote
% (Sept 2022, modified May 2023)
% y: measurement vector or matrix
% A: measurement matrix
% lambda_n: noise variance parameter
% suppsize: sparse support set size to be recovered
% Please cite: R. R. Pote and B. D. Rao.“Light-Weight Sequential SBL Algorithm: An Alternative to OMP”. In: 2023 IEEE
% International Conference on Acoustics, Speech and Signal Processing (ICASSP). 2023, pp. 1–5.


m=size(A,1);
n=size(A,2);
L=size(y,2);
% Initialization
qj=zeros(n,L);
qsj=zeros(n,1);
sj=zeros(n,1);
gamma_est=zeros(n,1);
kpset = zeros(suppsize,1);
Cinv=eye(m)/lambda_n;
CiA=Cinv*A;
xhat = zeros(n,L);
w_norm=zeros(suppsize,1);
w_prev1mat=zeros(m,suppsize);

% Main Loop
tic
for p=1:suppsize
    if p==1
        qj=A'*(y/lambda_n);
        qsj=sum(conj(qj).*qj,2);% abs(qj).^2;
        sj=(sum(conj(A).*A,1).')/lambda_n;%(m/lambda_n)*ones(n,1); %to exploit structure in A
    end
    if p==1
        [val,kp]=max(qsj);
        val=lambda_n*val/m;
    else
        [val,kp]=max(qsj./sj);
    end
    if val>1 % add new column to the model
        kpset(p)=kp;
        gamma_est(kp)=(val-1)/sj(kp);
        
        w_norm(p)=1/(1/gamma_est(kp)+sj(kp));
        w_prev1=CiA(:,kp)-w_prev1mat(:,1:p-1)*((w_prev1mat(:,1:p-1)'*A(:,kp)).*w_norm(1:p-1,1));
        w_prev1mat(:,p)=w_prev1;
        w_prev2tmp=(A'*w_prev1);
        w_prev2=w_prev2tmp*(w_norm(p)*qj(kp,:)); %(qj(kp)/w_norm)*w_prev1;
        qj=qj-w_prev2;% slight abuse of notation, for `j' in model qj (MVDR) is actually Qj (MPDR). For `j' in candidateset qj (MVDR) is correct
        qsj=sum(conj(qj).*qj,2);
        sj=sj-w_norm(p)*(conj(w_prev2tmp).*w_prev2tmp);%(1/(val-1))*(conj(w_prev2).*w_prev2);
    else
        warning('Seq. SBL did not add new column')
    end
end
xhat(kpset(kpset>0),:)=repmat(gamma_est(kpset(kpset>0)),1,L).*qj(kpset(kpset>0),:);
compute_time=toc;
end