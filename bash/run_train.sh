

export CUDA_VISIBLE_DEVICES=3
export PYTHONNOUSERSITE=1

cd ../experiment


# =========================
# Fixed configs
# =========================
text_model=roberta-base
audio_models=(wav2vec2_0 wavlm)
datasets=(iemocap6 meld7)

pooling_mode=curr_only
speaker_mode=self_other

num_epochs=20
learning_rate=2e-5
min_lr=5e-5

# SSL / loss configs
ssl_mode=supcon
weak_pos=0.3
plutchik_instance_coeff=0.5
plutchik_instance_match=True
plutchik_ipc_grad_start_ep=5
at_barlow_align=True
weight_decay=0.0001

# =========================
# Loop
# =========================
for dataset in "${datasets[@]}"; do

  if [[ "$dataset" == "iemocap6" ]]; then
    batch_size=64
    multimodal_pooling=cls
    max_txt_len=256
    split_data_dir_base=train_split/train_split_iemocap
  else
    batch_size=32
    multimodal_pooling=curr_only
    max_txt_len=128
    split_data_dir_base=train_split/train_split_meld
  fi

  split_data_dir=${split_data_dir_base}
  exp_dir=${dataset}_${multimodal_pooling}

  for audio_model in "${audio_models[@]}"; do

    echo "=============================================="
    echo "Dataset: $dataset | Audio: $audio_model"
    echo "Batch size: $batch_size | Pooling: $multimodal_pooling"
    echo "=============================================="

    python finetune.py \
      --dataset $dataset \
      --text_model $text_model \
      --audio_model $audio_model \
      --modal multimodal \
      --pooling_mode $pooling_mode \
      --multimodal_pooling $multimodal_pooling \
      --speaker_mode $speaker_mode \
      --split_data_dir $split_data_dir \
      --load_pt \
      --finetune_roberta True \
      --batch_size $batch_size \
      --max_txt_len $max_txt_len \
      --num_epochs $num_epochs \
      --learning_rate $learning_rate \
      --min_lr $min_lr \
      --weight_decay $weight_decay \
      --best_metric mf1 \
      --ssl_mode $ssl_mode \
      --weak_pos $weak_pos \
      --at_barlow_align $at_barlow_align \
      --plutchik_instance_coeff $plutchik_instance_coeff \
      --plutchik_instance_match $plutchik_instance_match \
      --plutchik_ipc_grad_start_ep $plutchik_ipc_grad_start_ep \
      --exp_dir $exp_dir

  done
done
