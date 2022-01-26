#best
# CUDA_VISIBLE_DEVICES=1 python test_image_STE.py --dataRoot '/home/ivdai/Temp2018/ZWP/data/1-train/dehw/dehw_testA_dataset/images'  \
#             --batchSize 1 \
#             --pretrain 'ckpts_str_m000_20_flex/STE_best_38.2469.pth' \
#             --savePath 'res/' \
#             --net 'str'

CUDA_VISIBLE_DEVICES=0 python test_image_STE.py --dataRoot '../data/dehw_testB_dataset'  \
            --batchSize 1 \
            --pretrain 'STE_best.pdparams' \
            --savePath 'res/' \
            --net 'mix'
