function skld_test = function_20150714_test_5(lambda,nbins,normtype)

x1 = load('x1.mat');
y1 = load('y1.mat');
x2 = load('x2.mat');
y2 = load('y2.mat');
x3 = load('x3.mat');
y3 = load('y3.mat');
x4 = load('x4.mat');
y4 = load('y4.mat');
x5 = load('x5.mat');
y5 = load('y5.mat');

cvx_setup;
cvx_solver Mosek;

X1 = x1.x1(:,1:nbins*3996);
X2 = x2.x2(:,1:nbins*3996);
X3 = x3.x3(:,1:nbins*3996);
X4 = x4.x4(:,1:nbins*3996);
X5 = x5.x5(:,1:nbins*3996);

Y1 = y1.y1;
Y2 = y2.y2;
Y3 = y3.y3;
Y4 = y4.y4;
Y5 = y5.y5;

% test 1
X_train = cat(1,X1,X2,X3,X4);
Y_train = cat(1,Y1,Y2,Y3,Y4);

X_test =  X5;
Y_test =  Y5;

A_train = X_train(:,1:999);
C_train = X_train(:,1000:1998);
G_train = X_train(:,1999:2997);
T_train = X_train(:,2998:3996);

A_test = X_test(:,1:999);
C_test = X_test(:,1000:1998);
G_test = X_test(:,1999:2997);
T_test = X_test(:,2998:3996);


if nbins>1
    for i = 1:nbins-1
        A_train = cat(2,A_train,X_train(:,i*3996+1:i*3996+999));
        C_train = cat(2,C_train,X_train(:,i*3996+1000:i*3996+1998));
        G_train = cat(2,G_train,X_train(:,i*3996+1999:i*3996+2997));
        T_train = cat(2,T_train,X_train(:,i*3996+2998:i*3996+3996));
        
        A_test = cat(2,A_test,X_test(:,i*3996+1:i*3996+999));
        C_test = cat(2,C_test,X_test(:,i*3996+1000:i*3996+1998));
        G_test = cat(2,G_test,X_test(:,i*3996+1999:i*3996+2997));
        T_test = cat(2,T_test,X_test(:,i*3996+2998:i*3996+3996));
    end  
end

ntrain = size(X_train,1);
ntest = size(X_test,1);

one = ones(ntrain,1);
A_train = cat(2,A_train,one);
C_train = cat(2,C_train,one);
G_train = cat(2,G_train,one);
T_train = cat(2,T_train,one);

one = ones(ntest,1);
A_test = cat(2,A_test,one);
C_test = cat(2,C_test,one);
G_test = cat(2,G_test,one);
T_test = cat(2,T_test,one);

YA_train = Y_train(:,1);
YC_train = Y_train(:,2);
YG_train = Y_train(:,3);
YT_train = Y_train(:,4);

YA_test = Y_test(:,1);
YC_test = Y_test(:,2);
YG_test = Y_test(:,3);
YT_test = Y_test(:,4);

cvx_expert true
cvx_begin
    variable W(nbins*999+1,1)

    AW_train = exp(A_train*W);
    CW_train = exp(C_train*W);
    GW_train = exp(G_train*W);
    TW_train = exp(T_train*W);

    XW_sum_train = AW_train + CW_train + GW_train + TW_train;
    PA_train = AW_train./XW_sum_train;
    PC_train = CW_train./XW_sum_train;
    PG_train = GW_train./XW_sum_train;
    PT_train = TW_train./XW_sum_train;

    %d1_train = PA_train.*log(PA_train./YA_train)+PC_train.*log(PC_train./YC_train)+PG_train.*log(PG_train./YG_train)+PT_train.*log(PT_train./YT_train);
    d2_train = YA_train.*log(YA_train./PA_train)+YC_train.*log(YC_train./PC_train)+YG_train.*log(YG_train./PG_train)+YT_train.*log(YT_train./PT_train);
    %kld1_train = sum(d1_train)/ntrain;
    kld2_train = sum(d2_train)/ntrain;
    %skld = kld1_train +kld2_train;
    penalty = norm(W,normtype);
    
        minimize(kld2_train + lambda*penalty)
    
cvx_end


kld2_train

% testing section

AW_test = exp(A_test*W);
CW_test = exp(C_test*W);
GW_test = exp(G_test*W);
TW_test = exp(T_test*W);
XW_sum_test = AW_test + CW_test + GW_test + TW_test;
PA_test = AW_test./XW_sum_test;
PC_test = CW_test./XW_sum_test;
PG_test = GW_test./XW_sum_test;
PT_test = TW_test./XW_sum_test;

d1_test = PA_test.*log(PA_test./YA_test)+PC_test.*log(PC_test./YC_test)+PG_test.*log(PG_test./YG_test)+PT_test.*log(PT_test./YT_test);
d2_test = YA_test.*log(YA_test./PA_test)+YC_test.*log(YC_test./PC_test)+YG_test.*log(YG_test./PG_test)+YT_test.*log(YT_test./PT_test);
kld1_test = sum(d1_test)/ntest
kld2_test = sum(d2_test)/ntest

skld_test = kld1_test +kld2_test

