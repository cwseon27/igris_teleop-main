# Robot Control Kinematics

This package is the canonical location for kinematics-related robot-control code.

- `joints.py`: joint and motor index definitions plus common joint groups.
- `pr2ab.py`: PR/PJS to AB/MS parallel-joint transforms.
- `fk/`: forward kinematics helpers.
- `ik/`: inverse kinematics solvers and IK environment setup.

The old `robot_control.controller.kinematics`, `robot_control.controller.pr2ab`, `robot_control.fk`, and `robot_control.ik` paths remain as compatibility wrappers.
