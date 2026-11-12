#!/usr/bin/env bash
# Reproduces every experiment in the paper. RAW must contain ML-EdgeIIoT-dataset.csv and wustl_iiot_2021.csv.
set -e
RAW=${1:-raw}; export PSI_OUT=${PSI_OUT:-out}; mkdir -p $PSI_OUT
python3 prep_data.py $RAW $PSI_OUT
# main benchmark (connection-disjoint, 5 folds)
python3 run_exp.py --data edge_audited --models lgbm --robust 1
python3 run_exp.py --data edge_audited --models psi+ple,inc1d,ftt+ple --robust 1
python3 run_exp.py --data edge_audited --models xgb,rf,lr,mlp,cnn1d,inc2d,cnnbigru --robust 0
python3 run_exp.py --data wustl --models lgbm,xgb,rf,lr --robust 0
python3 run_exp.py --data wustl --models lgbm --robust 1 --tag R
python3 run_exp.py --data wustl --models psi+ple,inc1d,ftt+ple --robust 1
python3 run_exp.py --data wustl --models mlp,cnn1d,inc2d,cnnbigru --robust 0
# feature view x split unit
python3 run_exp.py --data edge_ferrag --protocol random --models lgbm
python3 run_exp.py --data edge_ferrag --protocol group --models lgbm
python3 run_exp.py --data edge_audited --protocol random --models lgbm
for v in "edge_ferrag random" "edge_audited random" "edge_ferrag group"; do set -- $v; python3 run_exp.py --data $1 --protocol $2 --models psi+ple --folds 0,1; done
# leave-one-family-out (fold 1)
for h in Port_Scanning Password XSS DDoS_HTTP MITM Ransomware; do python3 run_exp.py --data edge_audited --models psi+ple --folds 0 --holdout $h; done
for h in Reconn CommInj; do python3 run_exp.py --data wustl --models psi+ple --folds 0 --holdout $h; done
# ablations (folds 1 and 2)
python3 run_exp.py --data edge_audited --models psi,psi+ple+nomask,psi+ple+noseg,psi+ple+nocross,psi+ple+randin,tokinc+ple,inc1d+perm11,inc1d+ser --folds 0,1
python3 timing.py
python3 make_all.py          # writes $PSI_OUT/V.json with every number used in the paper
python3 fig_arch.py; python3 fig_extra.py
