# Robot Control Assets

This directory is the canonical robot-description asset location for robot-control code.

- `urdf/` is used by FK, IK, calibration, and web visualization.
- `meshes/` contains the mesh files referenced by those URDF assets.
- `igris_teleop/sim/robot` keeps simulator-local MuJoCo XML/assets for runtime compatibility.

Do not add new canonical robot meshes under `sim/robot`; add them here first, then update simulator XML paths separately.
