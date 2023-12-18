function [ Aout, xt, error, time ] = SNMF_BPSG_SAGA_e(y,sr,n_epochs, tau, r, Ain, xin) 
%      argmin_{A,X} \|Y - AX\|_F^2 
%      s.t. \|A_k\|_0 <= tau \forall k, A_{i,j} >=0,  X_{i,j} >= 0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[n , d] =   size(y);
xi     =    zeros(r,d); 
m       =  floor( n / sr ); 
m2 = floor( d/sr );
grad_book   =   zeros(r , d, sr);
avg         =   sum(grad_book,3)./sr; % creat the average of the gradients
grad_book2   =   zeros(n , r, sr);
avg2         =   sum(grad_book2,3)./sr; % creat the average of the gradients
his         =  zeros(r, d,n_epochs);
error = zeros(n_epochs,1);
pn = 5; % number of power iterations
% initialization
A = Ain; 
xi = xin;
A_old = A;
xi_old = xi;
t=1;
pp=0.9;
norm_y = norm(y,'fro');
md = zeros(1,r);
time = zeros(n_epochs, 1);
t_total = 0;
e0 = 0.5 * ( norm( A_old * xi_old - y ,'fro') )^2 ;
for k = 1 : n_epochs % every epoch
    tic;    
    idxb  =  randperm(sr,sr);
    idxb2 =  randperm(sr,sr);
    t=t+1;
    tpp = t^pp;
    for i = 1 : sr         
        if idxb(i) == sr            
         idx    =  (1 + (idxb(i)-1)*m): n;%(idxb(i)*m);  
        else        
         idx    =  (1 + (idxb(i)-1)*m): (idxb(i)*m); %randperm(n, m);%        
        end                    
        As     =   A_old(idx,:);%.*sqrt(n/m);    
        ys     =   y(idx,:);%.*sqrt(n/m);        
        L_A = power_method(As, pn);   
        u  = 1/L_A; 
        u = min(u/tpp, u);
        grad   =   As'*(As*xi_old - ys); %stochastic gradient calculation       
        grad_diff = grad - grad_book(:, :,idxb(i)); 
        coeff = 3*(norm(A_old,'fro')^2+norm(xi_old,'fro')^2)+norm_y;
        if k > 1      
           xi = coeff*xi_old-(grad_diff + avg)/u;
        else
           xi = coeff*xi_old-grad/u;
        end    
        xi(xi < 0) = 0; % non-negative constraint       
        avg    =  avg + grad_diff./sr; % update the average gradient               
        grad_book(:,:,idxb(i)) = grad;            % update the gradient book                
        if idxb2(i) == sr           
         idx    =  (1 + (idxb2(i)-1)*m2): d;%(idxb(i)*m);  
        else       
         idx    =  (1 + (idxb2(i)-1)*m2): (idxb2(i)*m2); %randperm(n, m);%       
        end         
        xi2     =   xi_old(:,idx);%.*sqrt(n/m);
        ys2     =   y(:,idx);%.*sqrt(n/m);       
        L_x = power_method(xi2, pn);     
        uy = 1/L_x; 
        uy = min(uy/tpp, uy);
        grad2   = (xi2*(A*xi2 - ys2)')';  %(A*xi2 - ys2)*xi2';% 
        grad_diff2 = grad2 - grad_book2(:,:,idxb2(i));        
        if k > 1          
            A = coeff*A_old - (grad_diff2 + avg2)/uy;
        else          
            A = coeff*A_old - grad2/uy;
        end          
        B = sort(abs(A), 1, 'descend');
        md = B(tau,:);           
       for q = 1:r % hard - tresholding
            A(:,q) = wthresh(A(:,q),'h',md(q));        
       end  
        A(A<0) = 0;        % non-negative constraint  
        cor_r_3 = 3*(norm(A,'fro')^2+norm(xi,'fro')^2);
        r_sol = solve_eq_3(cor_r_3, 0, norm_y, -1);
        xi = r_sol*xi;
        A = r_sol*A;
        xi_old = xi;
        A_old = A;
        avg2    =  avg2 +  grad_diff2./sr; % update the average gradient              
        grad_book2(:,:,idxb2(i)) = grad2;            % update the gradient book 
    end   
    t1 = toc;
    t_total = t_total + t1;
    time(k) = t_total;    
    error(k) = 0.5 * ( norm( A * xi - y ,'fro') )^2 ;
end
xt = xi; % output
Aout = A;
error = [e0; error];

end







