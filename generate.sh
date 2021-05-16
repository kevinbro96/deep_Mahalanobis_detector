CUDA_VISIBLE_DEVICES=0 python ADV_Generate_LID_Mahalanobis_Subspace.py --dataset cifar10 --net_type resnet --adv_type FGSM --gpu 0 --outf ./data/adv_fullspace/
CUDA_VISIBLE_DEVICES=0 python ADV_Generate_LID_Mahalanobis_Subspace.py --dataset cifar10 --net_type resnet --adv_type BIM --gpu 0 --outf ./data/adv_fullspace/
CUDA_VISIBLE_DEVICES=0 python ADV_Generate_LID_Mahalanobis_Subspace.py --dataset cifar10 --net_type resnet --adv_type PGD --gpu 0 --outf ./data/adv_fullspace/
CUDA_VISIBLE_DEVICES=0 python ADV_Generate_LID_Mahalanobis_Subspace.py --dataset cifar10 --net_type resnet --adv_type CW --gpu 0 --outf ./data/adv_fullspace/



