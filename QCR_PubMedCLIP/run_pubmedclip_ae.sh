#!/bin/bash

###****RAD

#python main/main.py --cfg configs/qcr_pubmedclipRN50_ae_rad_16batchsize_withtfidf_nondeterministic.yaml
#python main/main.py --cfg configs/qcr_pubmedclipRN50x4_ae_rad_16batchsize_withtfidf_nondeterministic.yaml
#python main/main.py --cfg configs/qcr_pubmedclipViT_ae_rad_16batchsize_withtfidf_nondeterministic.yaml


###****SLAKE

#python main/main.py --cfg configs/qcr_pubmedclipRN50_ae_slake_16batchsize_withtfidf_nondeterministic.yaml
#python main/main.py --cfg configs/qcr_pubmedclipRN50x4_ae_slake_16batchsize_withtfidf_nondeterministic.yaml
#python main/main.py --cfg configs/qcr_pubmedclipViT_ae_slake_16batchsize_withtfidf_nondeterministic.yaml
#python main/main.py --cfg configs/qcr_slake_ViT_no_question.yaml --test TEST
#python main/main.py --cfg configs/qcr_slake_ViT_keywords_only.yaml --test TEST
#python main/main.py --cfg configs/qcr_slake_ViT_normal.yaml --test TEST
# python main/main.py --cfg configs/qcr_slake_ViT_keywords_and_normal.yaml --test TEST
# python main/main.py --cfg configs/qcr_slake_ViT_related_info.yaml
python main/main.py --cfg configs/qcr_slake_ViT_GCN.yaml