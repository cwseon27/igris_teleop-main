사용 예

# 여러 에피소드 한 번에 (기본)
python igris_teleop/training/analysis/dataset_analysis/trajectory_analysis.py \
  --dataset-root igris_teleop/dataset/AutoBagger \
  --episodes 0,1,2
# 에피소드별 개별 plot
python igris_teleop/training/analysis/dataset_analysis/trajectory_analysis.py \
  --dataset-root igris_teleop/dataset/AutoBagger \
  --episodes 0,1,2 \
  --separate
저장

# 합쳐서 저장 (trajectory_combined.png)
python igris_teleop/training/analysis/dataset_analysis/trajectory_analysis.py \
  --dataset-root igris_teleop/dataset/AutoBagger \
  --episodes 0,1,2 \
  --save ./traj_out


python igris_teleop/training/analysis/dataset_analysis/joint_analysis.py \
  --dataset-root igris_teleop/dataset/AutoBagger \
  --split train \
  --max-episodes 30 \
  --method softdtw \
  --gamma 1.0 \
  --band 0.1 \
  --resample-len 200
