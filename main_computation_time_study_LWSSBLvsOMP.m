% LWSeq. SBL vs Orthogonal Matching Pursuit: Array Processing Problem

clear all
addpath ./lib/
rng(1)
m = [15 19 20:20:200];
n = 200;
suppsizemax = 10;
datagen = 0;
ITER = 500;
SNR=30; % in dB
s_var=1;
w_var=s_var/10^(SNR/10);
sep=10; % make sure (max. sep X 2 X suppsizemax)<=n to ensure sep is always possible

% Grid Initialize
A = create_dictionary(m(end),n,'ULA');

seqsbl_l2error = zeros(ITER, length(m)); seqsbl_suppdist = zeros(ITER, length(m)); seqsbl_timecomp = zeros(ITER, length(m));
seqsbl_nvarNotgivenwvarby10_l2error = zeros(ITER, length(m)); seqsbl_nvarNotgivenwvarby10_suppdist = zeros(ITER, length(m)); seqsbl_nvarNotgivenwvarby10_timecomp = zeros(ITER, length(m));
seqsbl_nvarNotgivenwvar10_l2error = zeros(ITER, length(m)); seqsbl_nvarNotgivenwvar10_suppdist = zeros(ITER, length(m)); seqsbl_nvarNotgivenwvar10_timecomp = zeros(ITER, length(m));
redcompomp_l2error = zeros(ITER, length(m)); redcompomp_suppdist = zeros(ITER, length(m)); redcompomp_timecomp = zeros(ITER, length(m));
omp_l2error = zeros(ITER, length(m)); omp_suppdist = zeros(ITER, length(m)); omp_timecomp = zeros(ITER, length(m));

for iter=1:ITER
    iter
    switch datagen
        case 0
            nonzero_x = randn(suppsizemax,1)+1j*randn(suppsizemax,1);
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
    noise_vec=randn(m(end),1)+1j*randn(m(end),1);
    permute_n=randsample(n, n); % without replacement
    for m_iter=1:length(m)
        m_test=m(m_iter);
        A_test=A(1:m_test,:);
        for sep_iter=1:length(sep)
            [suppfull,loop_time] = min_separate(permute_n, suppsizemax, sep(sep_iter));%randsample(n, suppsizemax);
            %         loop_time
            xfull = zeros(n,1);
            xfull(suppfull)=nonzero_x;
            for isuppsize=suppsizemax:suppsizemax
                suppsize = isuppsize;
                supp = suppfull(1:suppsize);
                x = zeros(n,1);
                x(supp) = xfull(supp);
                y = sqrt(s_var/2)*A_test*x+sqrt(w_var/2)*noise_vec(1:m_test,:);
                
                %% LWS SBL (Lambda set to w_var)
                lambda_n=w_var;
                [xhat,kpset,gamma_est,compute_time]=LightWeightSeqSBL(y,A_test,lambda_n,suppsize);
                seqsbl_timecomp(iter,m_iter)=compute_time;
                seqsbl_l2error(iter,m_iter) = vecnorm(x-xhat)^2/vecnorm(x)^2;
                seqsbl_suppdist(iter,m_iter) = 1-length(intersect(supp,kpset))/max(suppsize,length(kpset));
                
                %% LWS SBL (Lambda set to w_var/10)
                lambda_n=w_var/10;
                [xhat,kpset,gamma_est,compute_time]=LightWeightSeqSBL(y,A_test,lambda_n,suppsize);
                seqsbl_nvarNotgivenwvarby10_timecomp(iter,m_iter)=compute_time;
                seqsbl_nvarNotgivenwvarby10_l2error(iter,m_iter) = vecnorm(x-xhat)^2/vecnorm(x)^2;
                seqsbl_nvarNotgivenwvarby10_suppdist(iter,m_iter) = 1-length(intersect(supp,kpset))/max(suppsize,length(kpset));
                
                %% LWS SBL (Lambda set to w_var*10)
                lambda_n=10*w_var;
                [xhat,kpset,gamma_est,compute_time]=LightWeightSeqSBL(y,A_test,lambda_n,suppsize);
                seqsbl_nvarNotgivenwvar10_timecomp(iter,m_iter)=compute_time;
                seqsbl_nvarNotgivenwvar10_l2error(iter,m_iter) = vecnorm(x-xhat)^2/vecnorm(x)^2;
                seqsbl_nvarNotgivenwvar10_suppdist(iter,m_iter) = 1-length(intersect(supp,kpset))/max(suppsize,length(kpset));
                
                %% 2.OMP
                
                [xhat,kpset,compute_time]=OMP_pseudoinvxhat(y,A_test,suppsize);
                omp_timecomp(iter,m_iter)=compute_time;
                omp_l2error(iter,m_iter) = vecnorm(x-xhat)^2/vecnorm(x)^2;
                omp_suppdist(iter,m_iter) = 1-length(intersect(supp,kpset))/max(suppsize,length(kpset));
                
                %% 2a.OMP (Reduced Complexity)
                
                [xhat,kpset,compute_time]=OMP_QRdecompxhat(y,A_test,suppsize);
                redcompomp_timecomp(iter,m_iter)=compute_time;
                redcompomp_l2error(iter,m_iter) = vecnorm(x-xhat)^2/vecnorm(x)^2;
                redcompomp_suppdist(iter,m_iter) = 1-length(intersect(supp,kpset))/max(suppsize,length(kpset));
                
            end
        end
    end
end

figure(1)
ax=plot(m, 1e3*mean(redcompomp_timecomp), '--rx', 'LineWidth', 2, 'MarkerSize',8);
hold on
plot(m, 1e3*mean(omp_timecomp), '-.ro', 'LineWidth', 2, 'MarkerSize',10)
plot(m, 1e3*mean(seqsbl_timecomp), '-bd', 'LineWidth', 2, 'MarkerSize',12)
plot(m, 1e3*mean(seqsbl_nvarNotgivenwvarby10_timecomp), '-.bs', 'LineWidth', 1, 'MarkerSize',18)
plot(m, 1e3*mean(seqsbl_nvarNotgivenwvar10_timecomp), '--bp', 'LineWidth', 1, 'MarkerSize',20)
xlabel('Measurement size (m)')
ylabel('Time in milliseconds')
legend('OMP-QR decomp.', 'OMP', 'LWS-SBL: \lambda=\sigma_n^2', 'LWS-SBL: \lambda=\sigma_n^2/10','LWS-SBL: \lambda=10*\sigma_n^2', 'Location', 'northwest')%, 'Seq. SBL-Target sj')
ax=ax.Parent;
set(ax, 'FontWeight', 'bold','FontSize',16)
xticks(m)
grid on

figure(2)
ax=plot(m, mean(redcompomp_l2error), '--rx', 'LineWidth', 2, 'MarkerSize',8);
hold on
plot(m, mean(omp_l2error), '-.ro', 'LineWidth', 2, 'MarkerSize',10)
plot(m, mean(seqsbl_l2error), '-bd', 'LineWidth', 2, 'MarkerSize',12)
plot(m, mean(seqsbl_nvarNotgivenwvarby10_l2error), '-.bs', 'LineWidth', 1, 'MarkerSize',18)
plot(m, mean(seqsbl_nvarNotgivenwvar10_l2error), '--bp', 'LineWidth', 1, 'MarkerSize',20)
xlabel('Measurement size (m)')
ylabel('Average relative L_2 error')
legend('OMP-QR decomp.', 'OMP', 'LWS-SBL: \lambda=\sigma_n^2', 'LWS-SBL: \lambda=\sigma_n^2/10','LWS-SBL: \lambda=10*\sigma_n^2', 'Location', 'northwest')
ax=ax.Parent;
set(ax, 'FontWeight', 'bold','FontSize',16)
xticks(m)
grid on

figure(3)
ax=plot(m, mean(redcompomp_suppdist), '--rx', 'LineWidth', 2, 'MarkerSize',8);
hold on
plot(m, mean(omp_suppdist), '-.ro', 'LineWidth', 2, 'MarkerSize',10)
plot(m, mean(seqsbl_suppdist), '-bd', 'LineWidth', 2, 'MarkerSize',12)
plot(m, mean(seqsbl_nvarNotgivenwvarby10_suppdist), '-.bs', 'LineWidth', 1, 'MarkerSize',18)
plot(m, mean(seqsbl_nvarNotgivenwvar10_suppdist), '--bp', 'LineWidth', 1, 'MarkerSize',20)
xlabel('Measurement size (m)')
ylabel('Probability of error in support')
legend('OMP-QR decomp.', 'OMP', 'LWS-SBL: \lambda=\sigma_n^2', 'LWS-SBL: \lambda=\sigma_n^2/10','LWS-SBL: \lambda=10*\sigma_n^2', 'Location', 'northwest')
ax=ax.Parent;
set(ax, 'FontWeight', 'bold','FontSize',16)
xticks(m)
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