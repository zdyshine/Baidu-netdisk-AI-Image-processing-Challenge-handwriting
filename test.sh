CUDA_VISIBLE_DEVICES=0 python test.py --dataRoot '../data/dehw_testB_dataset'  \
            --batchSize 1 \
            --pretrain 'STE_best.pdparams' \
            --savePath 'res/' \
            --net 'mix'
