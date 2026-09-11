# web_ui

브라우저 기반 UI입니다. 별도 프론트엔드 빌드 도구 없이 Python stdlib HTTP 서버와 정적 HTML/CSS/JS로 동작합니다.

## 구성

- `server.py`: UI HTTP server, 상태 API, worker control API를 제공합니다.
- `dataset_viewer_helper.py`: dataset viewer 탭에서 쓰는 dataset metadata helper입니다.
- `tunnels.py`: ngrok 같은 외부 tunnel 실행 보조입니다.
- `viser_bridge.py`: Viser 연동 보조 코드입니다.
- `static/index.html`: Web UI HTML entrypoint입니다.
- `static/app.js`: worker control, plot, camera preview, dataset viewer UI 로직입니다.
- `static/styles.css`: UI 스타일입니다.
- `static/vendor/echarts.min.js`: plot 렌더링용 vendored JS입니다.

기본 주소는 `http://127.0.0.1:8000/`입니다. 포트/호스트는 `--web-host`, `--web-port`, `IGRIS_WEB_HOST`, `IGRIS_WEB_PORT`로 바꿀 수 있습니다.
