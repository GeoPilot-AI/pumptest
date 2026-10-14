"""Array aliases shared by every module, so signatures stay readable."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
