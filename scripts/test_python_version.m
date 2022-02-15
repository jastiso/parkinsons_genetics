% load
fun_layer = load('/Volumes/bassett-data/Jeni/parkinsons_genes/data/subgrp_coexp/A_fn.csv');
gene_layer = load('/Volumes/bassett-data/Jeni/parkinsons_genes/data/subgrp_coexp/A_all.csv');
% run
T = 5; 
nt = 5001; 
[Ec1, xc1, vc1, optim_u1, eigs_c1] = duplex_cont_sol(fun_layer, gene_layer,T,nt,1,1); 
[Ec2, xc2, vc2, optim_u2,eigs_c2] = duplex_cont_sol(gene_layer, fun_layer,T,nt,1,1); 

% plot
figure(1); clf
scatter(1:194,Ec1); hold on
scatter(1:194,Ec2)

