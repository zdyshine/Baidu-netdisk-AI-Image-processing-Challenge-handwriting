# CUDA_VISIBLE_DEVICES=3 python3 train_STE.py --batchSize 4 \
#   --dataRoot '/home/ivdai/Temp2018/ZWP/data/1-train/dehw/dehw_train_dataset/images' \
#   --net 'str' \
#   --lr 1e-5 \
#   --modelsSavePath 'ckpts_str' \
#   --logPath 'logs'  \
#   --pretrained 'ckpts_str/STE_best_35.0386.pth' \
#   --mask_dir 'mask'


#CUDA_VISIBLE_DEVICES=0 python3 train_STE.py --batchSize 4 \
#  --dataRoot '/home/zhangdy/dataset/color_enhance/task2/dehw_train_dataset/images' \
#  --net 'str' \
#  --lr 1e-4 \
#  --modelsSavePath 'ckpts_str_m331_25' \
#  --logPath 'logs'  \
#  --mask_dir 'mask_331_25'


CUDA_VISIBLE_DEVICES=0 python3 train_STE.py --batchSize 4 \
  --dataRoot './dataset/task2/dehw_train_dataset/images' \
  --net 'idr' \
  --lr 1e-4 \
  --modelsSavePath 'ckpts_str_m331_25_idr' \
  --logPath 'logs'  \
  --mask_dir 'mask_331_25'
# CUDA_VISIBLE_DEVICES=0 python3 train_STE.py --batchSize 4 \
#   --dataRoot '/home/ivdai/Temp2018/ZWP/data/1-train/dehw/dehw_train_dataset/images' \
#   --net 'idr' \
#   --lr 1e-5 \
#   --modelsSavePath 'ckpts_idr_m000_5_l5' \
#   --logPath 'logs'  \
#   --pretrained 'ckpts_idr_m000_5/STE_best_37.0087.pth' \
#   --mask_dir 'mask_000_5'
