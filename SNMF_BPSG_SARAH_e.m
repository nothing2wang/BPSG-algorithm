function [ Aout, xt, error, time ] = SNMF_BPSG_SARAH_e(y,sr,n_epochs, tau, r, Ain, xin)
% Implement SPRING-SARAH for sparse non-negative matrix
% factorization
%      argmin_{A,X} \|Y - AX\|_F^2 
%
%      s.t. \|A_k\|_0 <= tau \forall k, A_{i,j} >=0,  X_{i,j} >= 0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[n , d] =   size(y);
m       =  floor( n / sr ); 
m2 = floor( d/sr );
pn = 5;
norm_y = norm(y,'fro');
error = zeros(n_epochs,1);
A = Ain; 
xi = xin;
At  =  A;
A_tmp = A;
xi_tmp = xi;
xt  = xi;
t = 1;
pp=1.2;
time = zeros(n_epochs, 1);
t_total = 0;
e0 = 0.5*(norm(A*xi-y,'fro'))^2 ;%+ tau * norm(xi(:), 1) + tau2 * norm(A(:), 1);
md = zeros(1,r);
for k = 1 : n_epochs % every epoch
    tic;
    t=t+1;
    tpp = t^pp;
    if k >-1%10  % full gradient at each outer loop      
        L_A = power_method(At, pn)/sr;
        u = 1/L_A;
        L_x = power_method(xt, pn)/sr;
        uy = 1/L_x; 
        u = min(u/tpp, u);
        uy = min(uy/tpp, uy);
        xi_tmp = xt;
        A_tmp = At;
        
        Atxty = At*xt-y;
        grad =  At'*Atxty/sr;
        grad2 = Atxty*xt'/sr;
        
        coeff = 3*(norm(At,'fro')^2+norm(xt,'fro')^2)+norm_y;
        xi = coeff*xt - grad/u;
        A = coeff*At - grad2/uy;
        %xi = xt - u * grad;
        xi(xi < 0) = 0;
        %A = At - uy * (grad2);
        B = sort(abs(A), 1, 'descend');
        md = B(tau,:);           
       for q = 1:r % hard - tresholding
            A(:,q) = wthresh(A(:,q),'h',md(q));        
       end 
        A(A<0) = 0;
    end
    cor_r_3 = 3*(norm(A,'fro')^2+norm(xi,'fro')^2);
    r_sol = solve_eq_3(cor_r_3, 0, norm_y, -1);
    xi = r_sol*xi;
    A = r_sol*A;
    
    idxb  =  randperm(sr,sr);
    idxb2 =  randperm(sr,sr);
    for i = 1 : sr  % inner loop            
        if idxb(i) == sr
            idx    =  (1 + (idxb(i)-1)*m): n;%(idxb(i)*m);
        else
            idx    =  (1 + (idxb(i)-1)*m): (idxb(i)*m); %randperm(n, m);%
        end
        As     =   A(idx,:);%.*sqrt(n/m);
        Ass    =   A_tmp(idx, :);
        ys     =   y(idx,:);%.*sqrt(n/m);
        if k <= 1%10
            grad   =  As'*(As*xi - ys); % SG for first few epochs
        else
            grad   =  As'*(As*xi - ys) - Ass'*(Ass*xi_tmp - ys) + grad;    % SARAH update
        end
        L_A = power_method(As, pn);%max(power_method(A(idx,:), pn) , L_A);
        u  = 1/L_A;
        u = min(u/tpp, u);
        if idxb2(i) == sr
            idx    =  (1 + (idxb2(i)-1)*m2): d;
        else
            idx    =  (1 + (idxb2(i)-1)*m2): (idxb2(i)*m2);
        end
        xi2     =   xi(:,idx);
        xii2    = xi_tmp(:,idx);
        ys2     =   y(:,idx);        
        if k <= 1%10
            grad2   =  (xi2*(A*xi2 - ys2)')' ; % SG for first few epochs
        else
            grad2   = (A*xi2 - ys2)*xi2' - (A_tmp*xii2 - ys2)*xii2' + grad2;%SARAH update
        end
        L_x = power_method(xi2, pn);
        uy = 1/L_x;
        uy = min(uy/tpp, uy);
        
        xi_tmp = xi;
        A_tmp = A;
        
        coeff = 3*(norm(A,'fro')^2+norm(xi,'fro')^2)+norm_y;
        xi     =   coeff*xi - grad/u;                  
        A = coeff*A - grad2/uy;
        
        xi(xi < 0) = 0;
        B = sort(abs(A), 1, 'descend');
        md = B(tau,:);
        for q = 1:r % hard - tresholding
            A(:,q) = wthresh(A(:,q),'h',md(q));
        end
        A(A<0) = 0;
        cor_r_3 = 3*(norm(A,'fro')^2+norm(xi,'fro')^2);
        r_sol = solve_eq_3(cor_r_3, 0, norm_y, -1);
        
        xi = r_sol*xi;
        A = r_sol*A;
    end  
    
    At  =  A;
    xt  = xi;
    t1 = toc;
    t_total = t_total + t1;
    time(k) = t_total;
    error(k) = 0.5 * ( norm( A * xi - y ,'fro') )^2;
end
xt = xi; % output
Aout = A;
error = [e0; error];
%err    =   [(norm(x_truth))^2, err];
end

%  v  = sign(c1 .* v) .* (max( abs(c1 .* v) -  (tau * (1/(3*L)) * c2) .* I_tmp, 0) );











