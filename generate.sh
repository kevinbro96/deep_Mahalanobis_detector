CUDA_VISIBLE_DEVICES=0 python ADV_Generate_LID_Mahalanobis_Subspace.py --dataset imagenet --net_type resnet --adv_type FGSM --gpu 0 --batch_size 8 --vae_path /public/data1/users/yangkaiwen8/auto_aug-master/results/main_imagenet_ce5/model_epoch60.pth --dataroot /public/data1/datasets/imagenet2012
#CUDA_VISIBLE_DEVICES=0 python ADV_Generate_LID_Mahalanobis_Subspace.py --dataset cifar10 --net_type resnet --adv_type BIM --gpu 0 --outf ./data/adv_fullspace/
#CUDA_VISIBLE_DEVICES=0 python ADV_Generate_LID_Mahalanobis_Subspace.py --dataset cifar10 --net_type resnet --adv_type PGD --gpu 0 --outf ./data/adv_fullspace/
#CUDA_VISIBLE_DEVICES=0 python ADV_Generate_LID_Mahalanobis_Subspace.py --dataset cifar10 --net_type resnet --adv_type CW --gpu 0 --outf ./data/adv_fullspace/



