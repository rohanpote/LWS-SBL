% LWSeq. SBL vs Orthogonal Matching Pursuit: Array Processing Problem

clear all
addpath ./lib/
rng(1)
m = 50;
n = 200;
L=[1:5 10 20:20:100 250 500 1000];
suppsizemax = 10;
datagen = 0;
ITER = 100;
SNR=30; % in dB
s_var=1;
w_var=s_var/10^(SNR/10);
sep=10; % make sure (max. sep X 2 X suppsizemax)<=n to ensure sep is always possible

% Grid Initialize
[A, u] = create_dictionary(m(end),n,'ULA');%randn(m,n);
% Anorm = A*diag(1./vecnorm(A));
% AtA = Anorm'*Anorm; AtA_tmp = abs(AtA);
% AtA_tmp(1:n+1:end)=0;
% mc = max(max(AtA_tmp));

seqsbl_l2error = zeros(ITER, length(L)); seqsbl_suppdist = zeros(ITER, length(L)); seqsbl_timecomp = zeros(ITER, length(L)); seqsbl_mse=zeros(ITER, length(L));

for iter=1:ITER
    iter
    switch datagen
        case 0
            nonzero_x = randn(suppsizemax,L(end))+1j*randn(suppsizemax,L(end));
        case 5
            nonzero_x = sqrt(3)*rand(suppsizemax,1).*(randn(suppsizemax,1)+1j*randn(suppsizemax,1));
        case 1
            nonzero_x = (rand(suppsizemax,1)+1).*(2*(rand(suppsizemax,1)>0.5)-1);
        case 2
            nonzero_x = randn(suppsizemax,1);
        case 3
            nonzero_x = 2*(rand(suppsizemax,1)>0.5)-1;
        case 4
            nonzero_x = trnd(1,suppsizemax,1);
    end
    noise_vec=randn(m(end),L(end))+1j*randn(m(end),L(end));
    permute_n=randsample(n, n); % without replacement
    u_perturb=(2*rand([1,suppsizemax])-1)/n;% off-grid perturb
    for l_iter=1:length(L)
        L_test=L(l_iter);
        for m_iter=1:length(m)
            m_test=m(m_iter);
            A_test=A(1:m_test,:);
            for sep_iter=1:length(sep)
                [suppfull,loop_time] = min_separate(permute_n, suppsizemax, sep(sep_iter));%randsample(n, suppsizemax);
                %         loop_time
                xfull = zeros(n,L_test);
                xfull(suppfull,:)=nonzero_x(:,1:L_test);
                for isuppsize=suppsizemax:suppsizemax
                    suppsize = isuppsize;
                    supp = suppfull(1:suppsize);
                    x = xfull;
                    u_actual=u; u_actual(supp)=u_actual(supp)+u_perturb;
                    A_actual=exp(-1j*pi*(0:m-1)'*u_actual);
                    y = sqrt(s_var/2)*A_actual*x+sqrt(w_var/2)*noise_vec(1:m_test,1:L_test);
                    
                    %% GridLess(GL)-LWS-SBL: LWS-SBL (Lambda set to w_var here)+Grid refinement
                    
                    % Dimension Reduction
                    [Q1, ~] = qr(y',0);
                    yred=y*Q1/sqrt(L_test); Lred = size(yred, 2);
                    lambda_n=w_var;
                    [xhat_norm,kpset,gamma_est,compute_time]=LightWeightSeqSBL(yred,A_test,lambda_n,suppsize);
                    
%                     u_grid_updated=u;
                    % Grid refinement
                    [gamma_est,u_grid_updated,Agrid_updated]=gridPtAdjPks(gamma_est,suppsize,u,A_test,(0:m_test-1),m_test,yred*yred',lambda_n);
                    
                    err_mat = repmat(u_grid_updated(kpset(kpset>0))', 1, suppsize)-repmat(u_actual(supp), length(u_grid_updated(kpset(kpset>0))), 1);
                    [err_vec, ind_vec] = min(abs(err_mat));
                    seqsbl_mse(iter,l_iter)=mean(err_vec.^2);                   
                end
            end
        end
    end
end

figure
load('Gridless_seq_SBL_ITER100inner10GtwoITER.mat')
loglog(L, sqrt(mean(seqsbl_mse)), '-ob', 'LineWidth', 2, 'MarkerSize',15);
hold on
clear all
load('Gridless_seq_SBL_ITER100inner10GoneITER.mat')
loglog(L, sqrt(mean(seqsbl_mse)), '-xb', 'LineWidth', 2, 'MarkerSize',10)
clear all
load('Gridless_seq_SBL_ITER100inner10GtenITER.mat')
loglog(L, sqrt(mean(seqsbl_mse)), '-sb', 'LineWidth', 1, 'MarkerSize',10)
clear all
load('Grid_seq_SBL_ITER100n200.mat')
loglog(L, sqrt(mean(seqsbl_mse)), '--dr', 'LineWidth', 2, 'MarkerSize',10)
clear all
load('Grid_seq_SBL_ITER100n2000sep100gridpts.mat')
loglog(L, sqrt(mean(seqsbl_mse)), '--pr', 'LineWidth', 2, 'MarkerSize',10)


% Stochastic CRB equal power sources assumed
Sigma_s=diag(sqrt(s_var))*eye(suppsizemax)*diag(sqrt(s_var))';
num_factor=w_var./(2*L);
Acrb = exp(-1j*pi*(0:m-1)'*u_actual(supp));
Dcrb = -1j*(0:m-1)'.*Acrb;
Ry_inv = eye(m)/((Acrb*Sigma_s*Acrb')+w_var*eye(m));
wonumfactor_crb_psi = eye(suppsizemax)/(real(Dcrb'*(eye(m)-((Acrb/(Acrb'*Acrb))*Acrb'))*Dcrb).*((Sigma_s*Acrb'*Ry_inv*Acrb*Sigma_s).'));
% in u space
sqrcrb_u_theta=real(sqrt(num_factor)*sqrt(mean(diag((1/pi)*wonumfactor_crb_psi*((1/pi).')))));

ax=loglog(L,sqrcrb_u_theta, '--k', 'LineWidth', 1);
legend({'Gridless LWS-SBL: ITER=2', 'Gridless LWS-SBL: ITER=1', 'Gridless LWS-SBL: ITER=10','LWS-SBL: n=200','LWS-SBL: n=2000','CRB'}, 'NumColumns',2)
ylabel('RMSE in u-space')
xlabel('Number of Snapshots (L)')
ax=ax.Parent;
set(ax, 'FontWeight', 'bold','FontSize',16)
xticks([1:5 10 20:20:60 100 250 500 1000])
grid on

%% Functions

function [suppfull,loop_time]=min_separate(permute_n, suppsizemax, sep)
supp_list=zeros(suppsizemax,1);
supp_list(1)=permute_n(1);
list_len=1;
loop_time=1;
while list_len<suppsizemax
    loop_time=loop_time+1;
    supp_tmp=permute_n(loop_time);
    if min(abs(supp_list-supp_tmp))>=sep
        list_len=list_len+1;
        supp_list(list_len)=supp_tmp;
    end
end
suppfull=supp_list;
end