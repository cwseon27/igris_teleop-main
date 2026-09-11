# policy_archive

이 디렉터리는 실행 시 필요한 학습 policy 파일을 repository 내부에 보관합니다.

## Reliability policy

`reliability/hand`와 `reliability/controller`에는 hybrid teleop의 tracking confidence
추론에 사용하는 RNN/HistGB joblib bundle이 있습니다.

- `hand_rnn.joblib`: HMD/OpenXR hand sequence 기반 hand confidence
- `hand_histgb.joblib`: hand RNN load 실패 시 fallback
- `controller_rnn.joblib`: chest controller motion sequence 기반 confidence
- `controller_histgb.joblib`: controller RNN load 실패 시 fallback

GUI가 Reliability를 시작할 때 기본 policy root는 이 디렉터리의 `reliability`입니다.
필요한 경우에만 `IGRIS_RELIABILITY_POLICY_ROOT`로 다른 경로를 지정합니다.

Policy inference 및 hand fusion 코드는 다음 repository-local ROS package에 있습니다.

- `ros_ws/src/igris_reliability_runtime`
- `ros_ws/src/mediapipe_hand_pose_bridge`
- `ros_ws/src/openxr_hand_to_igris_viewer`

파일 무결성은 다음 명령으로 확인합니다.

```bash
cd policy_archive
sha256sum -c SHA256SUMS
```
