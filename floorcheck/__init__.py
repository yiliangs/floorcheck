"""floorcheck: the D3 public checker.

A clean-room implementation of the transferable subset of the gate ladder,
written from the transfer specification alone under a clean-room rule. This
package contains no
code of the private solver, no FFI into it, no production constant, and no
firm-specific rule, and it imports nothing from `planaudit`.

Every constant in this package is a named value carrying a comment that cites the
specification section it came from. Where the specification was ambiguous, silent
or self-contradicting, the choice this implementation made is recorded where it is made rather than
decided in silence.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
