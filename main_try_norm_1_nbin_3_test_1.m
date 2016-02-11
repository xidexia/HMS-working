% parpool close force local
load('lambda.mat')
lambda1 = lambdaseq/4;
lambda = lambda1(1:5:50);

parpool('local',10)
skldout_norm_1_nbin_3_test_1 = ones(15,1);

c=1;
parfor i = 1:5
  skldout_norm_1_nbin_3_test_1(i,c)  =  function_20150714_test_1(lambda(i),3,1);
  
end
save skldout_norm_1_nbin_3_test_1
parpool close force local