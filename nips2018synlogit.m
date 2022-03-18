%% LAG - logistic regression real data
clear all
close all

%% data allocation 
[Xdata_28] = load('data2/data.txt'); 
[ydata_28] = load('data2/y.txt'); 
[Xdata_29] = load('data9/data.txt'); 
[ydata_29] = load('data9/y.txt');   
[Xdata_30] = load('data11/data.txt'); 
[ydata_30] = load('data11/y.txt');   

accuracy=1e-7;
num_iter=30000;
num_split=3;
num_workers=num_split*3;
X=cell(num_workers);
y=cell(num_workers);

num_feature=size(Xdata_29(1:50,:),2);
total_sample=size(Xdata_29(1:50,:),1); 
Xdata=randn(total_sample,num_feature);
ydata=[ydata_29(1:50)];

[Q R]=qr(Xdata); % qr分解 当特征维度过多时可通过矩阵的QR分解实现在尽可能保留原有信息的情况下降低维度
diagmatrix=diag(ones(total_sample,1)); % 创建对角矩阵
% [lambda]=eig(Xdata'*Xdata); % 特征值
Hmax=zeros(num_workers,1);
for i=1:num_workers
   X{i}=1^(i-1)*Q(:,i)*Q(:,i)'+diag(ones(total_sample,1)); % 数据预处理
   Hmax(i)=max(eig(X{i}'*X{i})); % 如果 A 是向量，则 max(A) 返回 A 的最大值。如果 A 为矩阵，则 max(A) 是包含每一列的最大值的行向量。 eig返回特征值列向量
   y{i}=ydata;
end

num_feature=size(X{1},2);

Hmax_sum=sum(Hmax); % 如果 A 是向量，则 sum(A) 返回元素之和。如果 A 是矩阵，则 sum(A) 将返回包含每列总和的行向量

%% data pre-analysis 数据预分析
lambda=0.001;
Hmax=zeros(num_workers,1);
for i=1:num_workers
   Hmax(i)=0.25*max(abs(eig(X{i}'*X{i})))+lambda; 
end
Hmax_sum=sum(Hmax);
hfun=Hmax_sum./Hmax; % A./B 表示 A 中元素与 B 中元素对应相除
nonprob=Hmax/Hmax_sum; 

Hmin=zeros(num_workers,1);
Hcond=zeros(num_workers,1);
for i=1:num_workers
   Hmin(i)=lambda; 
   Hcond(i)=Hmax(i)/Hmin(i);
end

X_fede=[];
y_fede=[];
for i=1:num_workers
  X_fede=[X_fede;X{i}];
  y_fede=[y_fede;y{i}];
end

triggerslot=10;
Hmaxall=0.25*max(eig(X_fede'*X_fede))+lambda;
% [f,x] = ecdf(y) returns the empirical cumulative distribution function f, evaluated at x, using the data in y.
[cdff,cdfx] = ecdf(Hmax*num_workers/Hmaxall); % 经验e累积分布函数cdf
comm_save=0;
for i=1:triggerslot % 触发器槽
    comm_save=comm_save+(1/i-1/(i+1))*cdff(find(cdfx>=min(max(cdfx),sqrt(1/(triggerslot*i))),1)); % k = find(X,n) 返回与 X 中的非零元素对应的前 n 个索引。列向量
end

heterconst=mean(exp(Hmax/Hmaxall)); % mean均值；Y = exp(X) 为数组 X 中的每个元素返回指数 e^x 
heterconst2=mean(Hmax/Hmaxall);
rate=1/(1+sum(Hmin)/(4*sum(Hmax)));

%% parameter initialization 参数初始化
%triggerslot=100;
theta=zeros(num_feature,num_iter);
grads=ones(num_feature,num_workers);
%stepsize=1/(num_workers*max(Hmax));
stepsize=1/Hmaxall;
thrd=10/(stepsize^2*num_workers^2)/triggerslot; % threshold?
comm_count=ones(num_workers,1);

theta2=zeros(num_feature,num_iter);
grads2=ones(num_feature,1);
stepsize2=stepsize;

theta3=zeros(num_feature,num_iter);
grads3=ones(num_feature,num_workers);
stepsize3=stepsize2/num_workers; % cyclic access learning

theta4=zeros(num_feature,num_iter);
grads4=ones(num_feature,num_workers);
stepsize4=stepsize/num_workers; % nonuniform-random access learning


thrd5=1/(stepsize^2*num_workers^2)/triggerslot;
theta5=zeros(num_feature,1);
grads5=ones(num_feature,num_workers);
stepsize5=stepsize;
comm_count5=ones(num_workers,1);

%thrd6=2/(stepsize*num_workers);
theta6=zeros(num_feature,1);
grads6=ones(num_feature,1);
stepsize6=0.5*stepsize;
comm_count6=ones(num_workers,1);


theta7=zeros(num_feature,1);
grads7=ones(num_feature,num_workers);
stepsize7=stepsize;
comm_count7=ones(num_workers,1);

% lambda=0.000;
%%  GD
comm_error2=[];
comm_grad2=[];
for iter=1:num_iter*10
    if mod(iter,1000)==0
        iter
    end
    % central server computation
    if iter>1
    grads2=-(X_fede'*(y_fede./(1+exp(y_fede.*(X_fede*theta2(:,iter))))))+num_workers*lambda*theta2(:,iter); % 新的本地梯度
        end
    % S = sum(A,dim) 沿维度 dim 返回总和。例如，如果 A 为矩阵，则 sum(A,2) 是包含每一行总和的列向量。 计算每列总和：S=sum(A)，返回行向量，或参数=1
    grad_error2(iter)=norm(sum(grads2,2),2); % 返回矩阵 X 的 p-范数，其中 p 为 1、2 或 Inf：如果 p = 2，则 n 近似于 max(svd(X))。这与 norm(X) 等效。可以认为是计算梯度的大小

    loss2(iter)=num_workers*lambda*0.5*norm(theta2(:,iter))^2+sum(log(1+exp(-y_fede.*(X_fede*theta2(:,iter))))); % 误差 loss function
    theta2(:,iter+1)=theta2(:,iter)-stepsize2*grads2; % 新的变量 利用梯度下降法更新
    comm_error2=[comm_error2;iter*num_workers,loss2(iter)]; % 记录？误差
    comm_grad2=[comm_grad2;iter*num_workers,grad_error2(iter)];  % 记录？梯度的范数（大小？）
end

for iter=i:num_iter
   if abs(loss2(iter)-loss2(end))<accuracy % 找到到达所需的精度目标的迭代轮次 --> 统计通信轮次 = 迭代次数 * 每轮迭代进行通信的计算节点个数 （GD中所有计算节点每轮都与server通信）
    fprintf('Communication rounds of GD\n');
       iter*num_workers  
       break
   end
end

%% LAG-PS
comm_iter=1;
comm_index=zeros(num_workers,num_iter);
comm_error=[];
comm_grad=[];
theta_temp=zeros(num_feature,num_workers);

for iter=1:num_iter*5
% 每轮迭代中
    comm_flag=0;
 %   local worker computation
    for i=1:num_workers  % 对于每个计算节点
        if iter>triggerslot
            trigger=0;
            for n=1:triggerslot
            trigger=trigger+norm(theta(:,iter-(n-1))-theta(:,iter-n),2)^2; % 公式15(b) 范数近似；新的变量theta由server发送给worker
            end

            if Hmax(i)^2*norm(theta_temp(:,i)-theta(:,iter),2)^2>thrd*trigger % thrd=threshold, 右边是触发通信的trigger，即梯度变化已经大到一定程度
                grads(:,i)=-(X{i}'*(y{i}./(1+exp(y{i}.*(X{i}*theta(:,iter))))))+lambda*theta(:,iter); % 计算新的本地梯度
                theta_temp(:,i)=theta(:,iter);
                comm_index(i,iter)=1;
                comm_count(i)=comm_count(i)+1; % 每个worker的迭代次数统计
                comm_iter=comm_iter+1; % 统计总通信迭代次数
                comm_flag=1;
            end
        end
    end
    
%    central server computation
    grad_error(iter)=norm(sum(grads,2),2);
    loss(iter)=num_workers*lambda*0.5*norm(theta(:,iter))^2+sum(log(1+exp(-y_fede.*(X_fede*theta(:,iter))))); % 误差计算
    theta(:,iter+1)=theta(:,iter)-stepsize*sum(grads,2); % 新的变量，通过梯度下降法更新
    

    if comm_flag==1
        comm_error=[comm_error;comm_iter,loss(iter)];
        comm_grad=[comm_grad;comm_iter,grad_error(iter)];
    elseif  mod(iter,1000)==0 % 强制进行通信？
        iter
        comm_iter=comm_iter+1;
        comm_error=[comm_error;comm_iter,loss(iter)];
        comm_grad=[comm_grad;comm_iter,grad_error(iter)];
    end
    if abs(loss(iter)-loss2(end))<accuracy % 通过与GD的loss相比较，找到达到目标精度所需的通信轮次，并退出循环
        fprintf('Communication rounds of LAG-PS\n');
        comm_iter
        break
    end
end

%% LAG-WK
comm_iter5=1;
comm_index5=zeros(num_workers,num_iter);
comm_error5=[];
comm_grad5=[];
for iter=1:num_iter*5

    comm_flag=0;
    % local worker computation
    for i=1:num_workers
        grad_temp=-(X{i}'*(y{i}./(1+exp(y{i}.*(X{i}*theta5(:,iter))))))+lambda*theta5(:,iter);
        if iter>triggerslot
            trigger=0;
            for n=1:triggerslot
            trigger=trigger+norm(theta5(:,iter-(n-1))-theta5(:,iter-n),2)^2;
            end

            if norm(grad_temp-grads5(:,i),2)^2>thrd5*trigger % 触发条件 worker收到来自server的theta后检查本地梯度的变化，并决定是否要通信
                grads5(:,i)=grad_temp;
                comm_count5(i)=comm_count5(i)+1;
                comm_index5(i,iter)=1;
                comm_iter5=comm_iter5+1; % 统计总通信轮次
                comm_flag=1;
            end
        end       
    end
    grad_error5(iter)=norm(sum(grads5,2),2);
    loss5(iter)=num_workers*lambda*0.5*norm(theta5(:,iter))^2+sum(log(1+exp(-y_fede.*(X_fede*theta5(:,iter))))); % 代价函数
    if comm_flag==1
       comm_error5=[comm_error5;comm_iter5,loss5(iter)]; 
       comm_grad5=[comm_grad5;comm_iter5,grad_error5(iter)]; 
    elseif  mod(iter,1000)==0
        iter
        comm_iter5=comm_iter5+1; 
        comm_error5=[comm_error5;comm_iter5,loss5(iter)]; 
       comm_grad5=[comm_grad5;comm_iter5,grad_error5(iter)]; 
    end
    theta5(:,iter+1)=theta5(:,iter)-stepsize5*sum(grads5,2); % 由server计算新的theta值
    if abs(loss5(iter)-loss2(end))<accuracy % 与GD相比是否到达所需精度，到达则停止迭代，退出循环
        fprintf('Communication rounds of LAG-WK\n');
        comm_iter5
        break
    end
end

%% cyclic IAG
for iter=1:num_iter*floor(num_workers)
    if mod(iter,100)==0
        iter
    end
    if iter>1
    % local worker computation
    i=mod(iter,num_workers)+1;
    grads3(:,i)=-(X{i}'*(y{i}./(1+exp(y{i}.*(X{i}*theta3(:,iter))))))+lambda*theta3(:,iter);
    end
    % central server computation
    grad_error3(iter)=norm(sum(grads3,2),2);
    loss3(iter)=num_workers*lambda*0.5*norm(theta3(:,iter))^2+sum(log(1+exp(-y_fede.*(X_fede*theta3(:,iter)))));
    theta3(:,iter+1)=theta3(:,iter)-stepsize3*sum(grads3,2);
    
    if abs(loss3(iter)-loss2(end))<accuracy
        fprintf('Communication rounds of Cyc-IAG\n');
        iter
        break
    end
end

%% non-uniform RANDOMIZED IAG
for iter=1:num_iter*floor(num_workers)
    if mod(iter,100)==0
        iter
    end
    % local worker computation
    workprob=rand;
    for i=1:num_workers
        if workprob<=sum(nonprob(1:i));
           break;
        end
    end
    %i=randi(num_workers);   
    if iter>1
    grads4(:,i)=-(X{i}'*(y{i}./(1+exp(y{i}.*(X{i}*theta4(:,iter))))))+lambda*theta4(:,iter);
    end
    % central server computation
    grad_error4(iter)=norm(sum(grads4,2),2);
    loss4(iter)=num_workers*lambda*0.5*norm(theta4(:,iter))^2+sum(log(1+exp(-y_fede.*(X_fede*theta4(:,iter)))));
    theta4(:,iter+1)=theta4(:,iter)-stepsize4*sum(grads4,2);
    
    if abs(loss4(iter)-loss2(end))<accuracy
        fprintf('Communication rounds of Num-IAG\n');
        iter
        break
    end
end




figure
semilogy(abs(loss3-loss2(end)),'k-','LineWidth',2);
hold on
semilogy(abs(loss4-loss2(end)),'g--','LineWidth',4);
hold on
semilogy(abs(loss-loss2(end)),'r-','LineWidth',2);
hold on
semilogy(abs(loss5-loss2(end)),'r--','LineWidth',2);
hold on
semilogy(abs(loss2-loss2(end)),'b-','LineWidth',2);
xlabel('Number of iteration','fontsize',16,'fontname','Times New Roman')
ylabel('Objective error','fontsize',16,'fontname','Times New Roman')
legend('Cyc-IAG','Num-IAG','LAG-PS','LAG-WK','Batch-GD')



figure
semilogy(abs(loss3-loss2(end)),'k-','LineWidth',2);
hold on
semilogy(abs(loss4-loss2(end)),'g--','LineWidth',4);
hold on
semilogy(comm_grad(:,1),abs(comm_error(:,2)-loss2(end)),'r-','LineWidth',2);
hold on
semilogy(comm_grad5(:,1),abs(comm_error5(:,2)-loss2(end)),'r--','LineWidth',2);
hold on
semilogy(comm_grad2(:,1),abs(comm_error2(:,2)-loss2(end)),'b-','LineWidth',2);
xlabel('Number of communications (uploads)','fontsize',16,'fontname','Times New Roman')
ylabel('Objective error','fontsize',16,'fontname','Times New Roman')
legend('Cyc-IAG','Num-IAG','LAG-PS','LAG-WK','Batch-GD')
