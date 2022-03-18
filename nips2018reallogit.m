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

accuracy=1e-8;
num_iter=4000;
num_split=6;
num_workers=num_split*3; % 每份数据分成split份分别处理，总共有3份数据，所以共需要这么多worker
X=cell(num_workers);
y=cell(num_workers);

% 分别处理3份数据，依次存入 X 和 y 中；对于每份数据，按批处理样本（split）
num_feature=min(min(size(Xdata_28,2),size(Xdata_29,2)),size(Xdata_30,2));
num_sample=size(Xdata_28,1);
per_split=floor(num_sample/num_split); % floor 向下取整

for n=1:num_split
X{n}=Xdata_28(per_split*(n-1)+1:per_split*n,1:num_feature); % 取出每个split所需要的数据 按行分出split（即按样本分批）
y{n}=ydata_28(per_split*(n-1)+1:per_split*n);
end

num_sample=size(Xdata_29,1);
per_split=floor(num_sample/num_split);
for n=1:num_split
X{n+num_split}=Xdata_29(per_split*(n-1)+1:per_split*n,1:num_feature);
y{n+num_split}=ydata_29(per_split*(n-1)+1:per_split*n);
end

num_sample=size(Xdata_30,1);
per_split=floor(num_sample/num_split);
for n=1:num_split
X{n+2*num_split}=Xdata_30(per_split*(n-1)+1:per_split*n,1:num_feature);
y{n+2*num_split}=ydata_30(per_split*(n-1)+1:per_split*n);
end


X_fede=[];
y_fede=[];
for i=1:num_workers
  X_fede=[X_fede;X{i}]; % 将所有的样本输入放入同一个矩阵中 本来是分别放在不同的cell中的
  y_fede=[y_fede;y{i}];
end

% [lambda]=eig(Xdata'*Xdata);
Hmax=zeros(num_workers,1);
% for i=1:num_workers
%    X{i}=1^(i-1)*Q(:,i)*Q(:,i)'+diag(ones(total_sample,1));
%    Hmax(i)=max(abs(eig(X{i}'*X{i}))); 
% end

for i=1:num_workers
   Hmax(i)=max(eig(X{i}'*X{i})); % 协方差矩阵 X和y的相关性
end
Hmax_sum=sum(Hmax);
hfun=Hmax_sum./Hmax;
nonprob=Hmax/Hmax_sum;

%% data pre-analysis
lambda=0.0001;
Hmax=zeros(num_workers,1);
for i=1:num_workers
   Hmax(i)=0.25*max(abs(eig(X{i}'*X{i})))+lambda; 
end
Hmax_sum=sum(Hmax);
hfun=Hmax_sum./Hmax;
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
[cdff,cdfx] = ecdf(Hmax*num_workers/Hmaxall);
comm_save=0;
for i=1:triggerslot
    comm_save=comm_save+(1/i-1/(i+1))*cdff(find(cdfx>=min(max(cdfx),sqrt(1/(triggerslot*i))),1));
end

heterconst=mean(exp(Hmax/Hmaxall));
heterconst2=mean(Hmax/Hmaxall);
rate=1/(1+sum(Hmin)/(4*sum(Hmax)));
%% parameter initialization
%triggerslot=100;
theta=zeros(num_feature,num_iter);
grads=ones(num_feature,num_workers);
%stepsize=1/(num_workers*max(Hmax));
stepsize=1/Hmaxall;
thrd=10/(stepsize^2*num_workers^2)/triggerslot;
comm_count=ones(num_workers,1);

theta2=zeros(num_feature,num_iter);
grads2=ones(num_feature,1);
stepsize2=stepsize;

theta3=zeros(num_feature,num_iter);
grads3=ones(num_feature,num_workers);
stepsize3=stepsize2/num_workers; % cyclic access learning

theta4=zeros(num_feature,num_iter);
grads4=ones(num_feature,num_workers);
stepsize4=0.5*stepsize/sqrt(num_workers); % nonuniform-random access learning


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
for iter=1:num_iter*2
    if mod(iter,100)==0
        iter
    end
    % central server computation
    if iter>1
    grads2=-(X_fede'*(y_fede./(1+exp(y_fede.*(X_fede*theta2(:,iter))))))+num_workers*lambda*theta2(:,iter); % logistic regression 梯度计算公式
        end
    grad_error2(iter)=norm(sum(grads2,2),2);

    loss2(iter)=num_workers*lambda*0.5*norm(theta2(:,iter))^2+sum(log(1+exp(-y_fede.*(X_fede*theta2(:,iter))))); % loss function
    theta2(:,iter+1)=theta2(:,iter)-stepsize2*grads2;
    comm_error2=[comm_error2;iter*num_workers,loss2(iter)]; % 列1：累积通信次数 后面：loss 每行记录一个loss及其对应的累计通信次数
    comm_grad2=[comm_grad2;iter*num_workers,grad_error2(iter)]; 
end

for iter=i:num_iter
   if abs(loss2(iter)-loss2(end))<accuracy
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
for iter=1:num_iter
    
    comm_flag=0;
    % local worker computation
    for i=1:num_workers
        if iter>triggerslot
            trigger=0;
            for n=1:triggerslot
            trigger=trigger+norm(theta(:,iter-(n-1))-theta(:,iter-n),2)^2;
            end
%             trigger=trigger/triggerslot;
            if Hmax(i)^2*norm(theta_temp(:,i)-theta(:,iter),2)^2>thrd*trigger % 通信触发的条件
                grads(:,i)=-(X{i}'*(y{i}./(1+exp(y{i}.*(X{i}*theta(:,iter))))))+lambda*theta(:,iter); % 满足条件，计算本地梯度
                theta_temp(:,i)=theta(:,iter);
                comm_index(i,iter)=1;
                comm_count(i)=comm_count(i)+1;
                comm_iter=comm_iter+1; % 计通信轮次
                comm_flag=1;
            end
        end
    end
    
    % central server computation
    grad_error(iter)=norm(sum(grads,2),2);
    loss(iter)=num_workers*lambda*0.5*norm(theta(:,iter))^2+sum(log(1+exp(-y_fede.*(X_fede*theta(:,iter))))); % 计算全局误差
    theta(:,iter+1)=theta(:,iter)-stepsize*sum(grads,2); % 计算下一步变量theta
    

    if comm_flag==1
        comm_error=[comm_error;comm_iter,loss(iter)]; % 列1：累积通信次数，其他列 loss；因为并不是每计算一次loss就腰通信一次，所以用这样的方法记录下来每个loss对应的累积通信次数，便于作图
        comm_grad=[comm_grad;comm_iter,grad_error(iter)];
    elseif  mod(iter,1000)==0
        iter
        comm_iter=comm_iter+1;
        comm_error=[comm_error;comm_iter,loss(iter)];
        comm_grad=[comm_grad;comm_iter,grad_error(iter)];
    end
    if abs(loss(iter)-loss2(end))<accuracy
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
for iter=1:num_iter

    comm_flag=0;
    % local worker computation
    for i=1:num_workers
        grad_temp=-(X{i}'*(y{i}./(1+exp(y{i}.*(X{i}*theta5(:,iter))))))+lambda*theta5(:,iter); % 计算本地梯度
        if iter>triggerslot
            trigger=0;
            for n=1:triggerslot
            trigger=trigger+norm(theta5(:,iter-(n-1))-theta5(:,iter-n),2)^2;
            end
%             trigger=trigger/triggerslot;
            if norm(grad_temp-grads5(:,i),2)^2>thrd5*trigger % 观察是否满足通信触发条件
                grads5(:,i)=grad_temp;
                comm_count5(i)=comm_count5(i)+1;
                comm_index5(i,iter)=1;
                comm_iter5=comm_iter5+1; % 计通信轮次，这里的通信轮次应该仅指worker向server上传的通信次数
                comm_flag=1;
            end
        end       
    end
    grad_error5(iter)=norm(sum(grads5,2),2);
    loss5(iter)=num_workers*lambda*0.5*norm(theta5(:,iter))^2+sum(log(1+exp(-y_fede.*(X_fede*theta5(:,iter))))); % 相当于server计算loss
    if comm_flag==1
       comm_error5=[comm_error5;comm_iter5,loss5(iter)]; 
       comm_grad5=[comm_grad5;comm_iter5,grad_error5(iter)]; 
    elseif  mod(iter,1000)==0
        iter
        comm_iter5=comm_iter5+1; 
        comm_error5=[comm_error5;comm_iter5,loss5(iter)]; 
       comm_grad5=[comm_grad5;comm_iter5,grad_error5(iter)]; 
    end
    theta5(:,iter+1)=theta5(:,iter)-stepsize5*sum(grads5,2); % 相当于server计算出下一步变量theta然后传给worker
    if abs(loss5(iter)-loss2(end))<accuracy
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

%% 
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
