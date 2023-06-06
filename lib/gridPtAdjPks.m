function [gamma_est,u_grid_updated,Agrid_updated]=gridPtAdjPks(gamma_est,K,u_grid_updated,Agrid_updated,spos,M,Ryoyo,lambda)
% gamma_est: solution from LWS-SBL
% K: number of source locations to be refined
% u_grid_updated: input grid gets modified in the code, only source
% locations are refined
% Agrid_updated: corresponding measurement matrix
% spos: sensor positions
% Ryoyo: sample covariance matrix
% lambda: noise variance parameter
% Please cite: R. R. Pote and B. D. Rao.“Light-Weight Sequential SBL Algorithm: An Alternative to OMP”. In: 2023 IEEE
% International Conference on Acoustics, Speech and Signal Processing (ICASSP). 2023, pp. 1–5.



% For rank-one update uncomment lines in the code; comment out lines that
% compute iSigma explicitly
% Sigma=Agrid_updated*diag(gamma_est)*Agrid_updated'+lambda*eye(M); % Sigma minus grid point i
% iSigma=eye(M)/Sigma;
for iterGdPtAdPks=1:2 % grid point adjustment around peaks iteration
    G=length(gamma_est); G_inner=10*G;
    [pks,locs]=findpeaks1(gamma_est);
    [mpks,mlocs]=maxk(pks, K);
    K_est=length(mlocs);
    m_candidates=sort(locs(mlocs));% one source
    u_est=u_grid_updated(m_candidates);
    for iterK=1:K_est
        m_iterK=m_candidates(iterK);
        if m_iterK>1; left_delta=u_grid_updated(m_iterK)-u_grid_updated(m_iterK-1); else; left_delta=u_grid_updated(m_iterK)+1; end
        if m_iterK<G; right_delta=u_grid_updated(m_iterK+1)-u_grid_updated(m_iterK); else; right_delta=1-u_grid_updated(m_iterK); end
        delta=left_delta/2+right_delta/2; resSeqSBL=delta/G_inner;
        u_candidates=linspace(u_est(iterK)-left_delta/2,u_est(iterK),floor(left_delta/2/resSeqSBL+1));
        u_candidates=union(u_candidates,linspace(u_est(iterK),u_est(iterK)+right_delta/2,floor(right_delta/2/resSeqSBL+1)));
        Gp=length(u_candidates);
        gamma_updated=zeros(1,Gp);
        I_gamma_opt=zeros(1,Gp);
        
        % Rank one update
%         wi_vec_remove=iSigma*Agrid_updated(:,m_iterK);
%         iSigma_mi=iSigma-wi_vec_remove*wi_vec_remove'/(-1/gamma_est(m_iterK)+Agrid_updated(:,m_iterK)'*wi_vec_remove);
        cmpgdind=setdiff(m_candidates,m_iterK); %setdiff(1:G,m_iterK); % complement set of grid index
        Sigma_mi=Agrid_updated(:,cmpgdind)*diag(gamma_est(cmpgdind))*Agrid_updated(:,cmpgdind)'+lambda*eye(M); % Sigma minus grid point i
        iSigma_mi=eye(M)/Sigma_mi;
        
        Aadpt_grid=exp(-1j*pi*spos'*u_candidates);
        q_i_sq=real(sum(conj(iSigma_mi*Aadpt_grid).*((Ryoyo*iSigma_mi)*Aadpt_grid)));
        s_i=real(sum(conj(Aadpt_grid).*(iSigma_mi*Aadpt_grid)));
        q_sq_by_s_i=q_i_sq./s_i;
        gamma_updated(q_i_sq>s_i)=(q_sq_by_s_i(q_i_sq>s_i)-1)./s_i(q_i_sq>s_i);
        I_gamma_opt(q_i_sq>s_i)=log(q_sq_by_s_i(q_i_sq>s_i))-q_sq_by_s_i(q_i_sq>s_i)+1;
        [valneigh,indneigh]=min(I_gamma_opt);
        if valneigh<0
            gamma_est(m_iterK)=gamma_updated(indneigh);
            u_grid_updated(m_iterK)=u_candidates(indneigh);
            Agrid_updated(:,m_iterK)=exp(-1j*pi*spos'*u_candidates(indneigh));
%             wi_vec_add=iSigma_mi*Agrid_updated(:,m_iterK);
%             iSigma=iSigma_mi-wi_vec_add*wi_vec_add'/(1/gamma_est(m_iterK)+Agrid_updated(:,m_iterK)'*wi_vec_add);
        end
    end
end
end

function [PKS,LOCS]=findpeaks1(Y) % assumes Y is a non-negative vector
diff1=diff([0 reshape(Y,1,[]) 0]);
LOCS=find(diff1(1:end-1)>0 & diff1(2:end)<0);
PKS=Y(LOCS);
end