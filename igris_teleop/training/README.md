# training

LeRobot dataset 처리, 학습, inference 테스트, dataset 분석 유틸을 모아둔 폴더입니다.

## 주요 파일

- `train.py`: LeRobot policy 학습 entrypoint입니다.
- `train_use_amp.py`: AMP 사용 학습 entrypoint입니다.
- `inference_test.py`: 저장된 dataset/checkpoint 기반 inference 테스트입니다.
- `inference_live_plot.py`: inference 결과를 실시간 plot으로 확인하는 테스트입니다.
- `dataset_utils.py`: dataset 경로와 metadata helper입니다.
- `dataset_manufacturing.py`, `merge_local_dataset.py`: local dataset 가공/병합 유틸입니다.
- `image_transforms.py`: 학습/추론용 image transform helper입니다.
- `loss_logger.py`: loss logging helper입니다.
- `datasets/`: segment labeling, init state 등 dataset 보조 도구입니다.
- `analysis/`: dataset 분석 스크립트와 분석 결과 산출물 위치입니다.

ML 환경은 `.venv-ml`을 사용합니다. 설치는 `./install_igris_teleop.sh` 또는 `./igris_teleop/setup_ml_venv.sh`로 수행합니다.

## 예시

```bash
python igris_teleop/training/train.py
python igris_teleop/training/inference_test.py --dataset-root <dataset> --episode 0 --checkpoint <checkpoint>
python igris_teleop/training/inference_live_plot.py --dataset-root <dataset> --episode 0 --checkpoint <checkpoint> --hz 30
```
