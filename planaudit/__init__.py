"""planaudit: room-polygon floor plans and the audit that grades them.

The package holds the plan record, its dataset loaders, and the audit over them:

- `planaudit.plan`      the dataset-neutral Plan record.
- `planaudit.datasets`  loaders that turn RPLAN, MSD and generated samples into Plan records.
"""

from planaudit.plan import AccessEdge, Plan, Room

__all__ = ["Plan", "Room", "AccessEdge"]
__version__ = "0.1.0"
