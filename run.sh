CUDA_VISIBLE_DEVICES=2,3  python train.py data-bin/wmt16_en_de_bpe32k \
        --arch transformer_wmt_en_de --share-all-embeddings \
          --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
            --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
              --lr 0.0007 --min-lr 1e-09 \
             --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0\
              --max-tokens  4096   --save-dir checkpoints/en-de\
              --update-freq 2 --no-progress-bar --log-format json --log-interval 50\
             --save-interval-updates  1000 --keep-interval-updates 20

----

python scripts/average_checkpoints.py --inputs /path/to/checkpoint/dir --num-epoch-checkpoints 10 --output averaged_model.pt

---- 


model=model.pt
subset="test"
  
CUDA_VISIBLE_DEVICES=3 python generate.py data-bin/wmt16_en_de_bpe32k  \
      --path checkpoints/$model --gen-subset $subset\
        --beam 4 --batch-size 128 --remove-bpe  --lenpen 0.6


----

utils/get_ende_bleu.sh checkpoints/model.pt
