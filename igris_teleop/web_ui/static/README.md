# web_ui/static

브라우저에서 직접 로드되는 정적 Web UI 파일입니다.

- `index.html`: UI entrypoint입니다.
- `app.js`: worker control, camera preview, plot, dataset viewer 로직입니다.
- `styles.css`: UI 스타일입니다.
- `vendor/`: vendored frontend dependency입니다.

별도 build step 없이 `web_ui/server.py`가 이 폴더를 정적으로 제공합니다.
