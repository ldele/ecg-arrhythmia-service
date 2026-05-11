"""AAMI beat class mapping for MIT-BIH annotations.

Reference: ANSI/AAMI EC57:2012. Maps the ~20 MIT-BIH beat symbols
to 5 superclasses: N, S, V, F, Q.
"""
from __future__ import annotations

# AAMI superclass for each MIT-BIH beat annotation symbol.
# Symbols not in this dict are non-beat annotations (rhythm markers,
# signal quality flags, etc.) and should be ignored.
AAMI_MAPPING: dict[str, str] = {
    # N: Normal and bundle branch block beats
    "N": "N",  # Normal
    "L": "N",  # Left bundle branch block
    "R": "N",  # Right bundle branch block
    "e": "N",  # Atrial escape
    "j": "N",  # Nodal (junctional) escape
    # S: Supraventricular ectopic beats
    "A": "S",  # Atrial premature
    "a": "S",  # Aberrated atrial premature
    "J": "S",  # Nodal (junctional) premature
    "S": "S",  # Supraventricular premature
    # V: Ventricular ectopic beats
    "V": "V",  # Premature ventricular contraction
    "E": "V",  # Ventricular escape
    # F: Fusion beats
    "F": "F",  # Fusion of ventricular and normal
    # Q: Unknown beats
    "/": "Q",  # Paced
    "f": "Q",  # Fusion of paced and normal
    "Q": "Q",  # Unclassifiable
}

AAMI_CLASSES: tuple[str, ...] = ("N", "S", "V", "F", "Q")


def map_to_aami(symbol: str) -> str | None:
    """Map a MIT-BIH beat symbol to its AAMI superclass.

    Returns None for non-beat annotations (e.g. '+', '~', '|').
    """
    return AAMI_MAPPING.get(symbol)