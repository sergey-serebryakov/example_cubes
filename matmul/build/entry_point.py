""" Example entry point script compatible with MLCube protocol. """
import numpy as np
import argparse
from enum import Enum
from typing import List


class Task(str, Enum):
    """ Every task has a name. This example defines two tasks - `task_a` and `task_b`.  """
    MATMUL = 'matmul'


def matmul(task_args: List[str]) -> None:
    # Parse task-specific command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, default=None, help="Output file.")
    args = parser.parse_args(args=task_args)

    # Implement this task here
    a = np.array([[5, 1, 3], [1, 1, 1], [1, 2, 1]])
    b = np.array([1, 2, 3])
    c = a.dot(b)
    np.savetxt(args.output_file, c)


def main():
    # Every MLCube runner passes a task name as the first argument. Other arguments are task-specific.
    parser = argparse.ArgumentParser()
    parser.add_argument('mlcube_task', type=str, help="Task for this MLCube.")

    # The `mlcube_args` contains task name (mlcube_args.mlcube_task)
    # The `task_args` list needs to be parsed later when task name is known
    mlcube_args, task_args = parser.parse_known_args()

    if mlcube_args.mlcube_task == Task.MATMUL:
        matmul(task_args)
    else:
        raise ValueError(f"Unknown task: '{mlcube_args.mlcube_task}'")


if __name__ == '__main__':
    main()
