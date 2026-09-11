from __future__ import annotations

from dataclasses import asdict, dataclass


EXPERIMENT_GEOM_PREFIX = "exp_task"
EXPERIMENT_VISUAL_ONLY_SUFFIX = "_marker"


@dataclass(frozen=True)
class ExperimentTask:
    task_id: int
    key: str
    name: str
    workspace: str
    objects: str
    goal: str

    def to_dict(self) -> dict[str, str | int]:
        return asdict(self)


EXPERIMENT_TASKS: tuple[ExperimentTask, ...] = (
    ExperimentTask(
        task_id=1,
        key="small_box_pnp",
        name="Task 1 - Small Box PnP",
        workspace="Front table 1600 x 600 x 800 mm; right tray 400 x 300 x 100 mm",
        objects="Two 50 x 50 x 50 mm boxes on the robot-left side",
        goal="Move the small boxes into the right tray",
    ),
    ExperimentTask(
        task_id=2,
        key="cube_stacking",
        name="Task 2 - Cube Stacking",
        workspace="Front table 1600 x 600 x 800 mm",
        objects="Two 50 x 50 x 50 mm cubes placed side by side",
        goal="Pick up one cube and stack it on the other cube",
    ),
    ExperimentTask(
        task_id=3,
        key="large_box_transfer",
        name="Task 3 - Large Box Transfer",
        workspace="Two 600 x 600 x 800 mm tables at robot-left/right 45 degrees",
        objects="One 400 x 300 x 300 mm box, mass 300 g, high-friction surface",
        goal="Move the box from the left table to the right table",
    ),
    ExperimentTask(
        task_id=4,
        key="peg_in_hole",
        name="Task 4 - Peg in Hole",
        workspace="Front table 1600 x 600 x 800 mm with a raised socket",
        objects="One 30 mm diameter, 120 mm long peg and a 40 mm square hole",
        goal="Pick up the peg and insert it vertically into the hole",
    ),
)

EXPERIMENT_TASK_BY_ID = {task.task_id: task for task in EXPERIMENT_TASKS}
EXPERIMENT_TASK_IDS = frozenset(EXPERIMENT_TASK_BY_ID)


def experiment_task_id_from_name(name: str | None) -> int | None:
    value = str(name or "")
    for task_id in EXPERIMENT_TASK_IDS:
        if value.startswith(f"{EXPERIMENT_GEOM_PREFIX}{task_id}_"):
            return task_id
    return None


def experiment_tasks_payload() -> list[dict[str, str | int]]:
    return [task.to_dict() for task in EXPERIMENT_TASKS]
