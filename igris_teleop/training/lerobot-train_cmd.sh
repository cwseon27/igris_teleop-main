# ACT
lerobot-train \
  --output_dir ./outputs/ACT_AutoBagger_5090_$(date +%Y%m%d_%H%M%S) \
  --steps 500000 \
  --batch_size 16 \
  --num_workers 12 \
  --save_freq 20000 \
  \
  --dataset.root /home/dam2/dw_ws/igris/dataset/AutoBagger \
  --dataset.repo_id IGRIS_C \
  --dataset.image_transforms.enable true \
  \
  --policy.type act \
  --policy.repo_id IGRIS_C \
  --policy.chunk_size 100 \
  --policy.push_to_hub false \
  \
  --wandb.enable true \
  --wandb.project ACT_AutoBagger_5090_$(date +%Y%m%d_%H%M%S)


lerobot-train \
  --config_path=/home/dam2/dw_ws/igris/outputs/ACT_AutoBagger_5090_20260213_052656/checkpoints/last/pretrained_model/train_config.json \
  --resume=true \
  --steps 1000000 \
  --save_freq 10000





# convert dataset v21 to v30
python /home/dam2/dw_ws/igris/train/convert_dataset_v21_to_v30.py \
  --repo-id=ai_worker \
  --root=/home/dam2/dw_ws/ai_worker/dataset/ai_worker/ffw_sg2_rev1_bin_picking_v2 \
  --data-file-size-in-m=10000 \
  --video-file-size-in-mb=20000 \
  --push-to-hub=false


# convert dataset v30 to v21
python /home/dam2/dw_ws/igris/train/convert_v3_to_v2.py \
  --root=/home/dam2/dw_ws/ai_worker/dataset/ai_worker/ffw_sg2_rev1_bin_picking_v3_resize
  
  




pip install torchcodec==0.5 --index-url=https://download.pytorch.org/whl/cu128




python -m lerobot.scripts.train \
  --output_dir /root/ros2_ws/src/physical_ai_tools/lerobot/outputs/train/ACT_push_button_$(date +%Y%m%d_%H%M%S) \
  --steps 200000 \
  --batch_size 16 \
  --num_workers 12 \
  --save_freq 20000 \
  \
  --dataset.root /root/.cache/huggingface/lerobot/ai_worker/ffw_sg2_rev1_push_button \
  --dataset.repo_id ai_worker \
  --dataset.image_transforms.enable true \
  \
  --policy.type act \
  --policy.repo_id ai_worker \
  --policy.chunk_size 100 \
  --policy.push_to_hub false \
  \
  --wandb.enable true \
  --wandb.project ACT_push_button_$(date +%Y%m%d_%H%M%S)









lerobot-dataset-viz \
    --repo-id IGRIS_C \
    --root /home/dam2/dw_ws/igris/dataset/AutoBagger \
    --mode local \
    --episode-index 0




lerobot-train \
  --output_dir ./outputs/ACT_padding_5090_$(date +%Y%m%d_%H%M%S) \
  --steps 500000 \
  --batch_size 16 \
  --num_workers 12 \
  --save_freq 20000 \
  \
  --dataset.root /home/dam2/dw_ws/igris/dataset/AutoBagger \
  --dataset.repo_id IGRIS_C \
  --dataset.image_transforms.enable true \
  \
  --policy.type act \
  --policy.repo_id IGRIS_C \
  --policy.chunk_size 100 \
  --policy.push_to_hub false \
  \
  --wandb.enable true \
  --wandb.project ACT_AutoBagger_5090_$(date +%Y%m%d_%H%M%S)




# ACT
lerobot-train \
  --output_dir ./outputs/ACT_0304_throwing_5090_$(date +%Y%m%d_%H%M%S) \
  --steps 200000 \
  --batch_size 16 \ # 컴퓨터 사양에 따라 조절
  --num_workers 12 \ # 컴퓨터 사양에 따라 조절
  --save_freq 20000 \ 
  \
  --dataset.root /home/dam2/dw_ws/igris/dataset/0304_실증과제_dataset_train/0304_throwing \
  --dataset.repo_id IGRIS_C \
  --dataset.image_transforms.enable true \
  \
  --policy.type act \
  --policy.repo_id IGRIS_C \
  --policy.chunk_size 100 \
  --policy.push_to_hub false \
  \
  --wandb.enable true \ # wandb login 필요
  --wandb.project ACT_0304_throwing_5090_$(date +%Y%m%d_%H%M%S)

