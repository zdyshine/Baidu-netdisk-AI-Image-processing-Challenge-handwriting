CUDA_VISIBLE_DEVICES=0 python3 train.py --batchSize 4 \
  --dataRoot './dataset/task2/dehw_train_dataset/images' \
  --net 'idr' \
  --lr 1e-4 \
  --modelsSavePath 'ckpts_str_m331_25_idr' \
  --logPath 'logs'  \
  --mask_dir 'mask_331_25'
