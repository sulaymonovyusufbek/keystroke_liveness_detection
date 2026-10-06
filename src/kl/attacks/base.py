"""Attacker interface: a generator of machine-typed sessions in the Aalto schema."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

AALTO_COLUMNS: list[str] = [
    "PARTICIPANT_ID", "TEST_SECTION_ID", "SENTENCE", "USER_INPUT", "KEYSTROKE_ID",
    "PRESS_TIME", "RELEASE_TIME", "LETTER", "KEYCODE",
]


class Attacker(ABC):
    """Base class of all synthetic typing attacks.

    ``selection.py`` refuses to touch anything that is an :class:`Attacker` (or attack-labelled
    data); attackers are only used for evaluation (Phase 1: step 07 and the legacy replica).
    """

    #: short identifier used in result files
    name: str = "attacker"
    #: marker inspected by the attack-free selection guard
    is_attack: bool = True

    @abstractmethod
    def generate(self, sentence: str, rng: np.random.Generator) -> pd.DataFrame:
        """Return one machine-typed session for ``sentence`` in the Aalto schema.

        ``PARTICIPANT_ID`` / ``TEST_SECTION_ID`` are placeholders (-1); the caller overwrites them.
        All randomness must come from ``rng``.
        """
