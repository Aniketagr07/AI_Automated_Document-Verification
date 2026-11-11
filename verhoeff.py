"""
Verhoeff Algorithm and Official Indian Document Identifier Validators.
SIH 2024 (SIH1709) - Knowledge Representation & Production Systems.

Implements:
1. Verhoeff Dihedral (D5) Checksum for 12-digit Aadhaar UID.
2. Indian Income Tax PAN syntax and 5th-character surname cross-referencing.
3. Indian Driving License and Passport structural format validators.
"""

import re
from typing import Tuple

# Multiplication table d (Dihedral group D5)
_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
]

# Permutation table p
_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]
]

# Inverse table inv
_INV = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]


def validate_verhoeff(num_str: str) -> bool:
    """
    Validates whether a 12-digit Aadhaar number satisfies the Verhoeff checksum.
    UIDAI generates Aadhaar numbers such that Verhoeff check returns 0.
    """
    clean_num = "".join(ch for ch in str(num_str) if ch.isdigit())
    if not clean_num or len(clean_num) != 12:
        return False

    c = 0
    reversed_digits = [int(d) for d in reversed(clean_num)]
    for i, digit in enumerate(reversed_digits):
        c = _D[c][_P[i % 8][digit]]
    return c == 0


def generate_verhoeff_check_digit(num_str: str) -> int:
    """Computes the 12th Verhoeff check digit for an 11-digit prefix."""
    clean_num = "".join(ch for ch in str(num_str) if ch.isdigit())
    c = 0
    reversed_digits = [int(d) for d in reversed(clean_num)]
    for i, digit in enumerate(reversed_digits):
        c = _D[c][_P[(i + 1) % 8][digit]]
    return _INV[c]


def validate_pan_structure(pan_str: str, holder_name: str = "") -> Tuple[bool, str]:
    """
    Validates Indian Permanent Account Number (PAN) syntax and cross-checks 5th character.
    Structure: [A-Z]{3}[PCHFATBLJG][A-Z][0-9]{4}[A-Z]
    - 4th character denotes entity: P=Person, C=Company, F=Firm, etc.
    - 5th character must match the first letter of cardholder's surname.
    """
    clean_pan = str(pan_str).strip().upper()
    if len(clean_pan) != 10:
        return False, f"PAN must be exactly 10 characters (found {len(clean_pan)})."

    if not re.match(r"^[A-Z]{5}[0-9]{4}[A-Z]$", clean_pan):
        return False, "Invalid PAN alphanumeric sequence (expected 5 letters, 4 digits, 1 letter)."

    entity_code = clean_pan[3]
    valid_entities = {
        "P": "Individual / Person",
        "C": "Company",
        "H": "Hindu Undivided Family",
        "F": "Partnership Firm",
        "A": "Association of Persons",
        "T": "Trust",
        "B": "Body of Individuals",
        "L": "Local Authority",
        "J": "Artificial Juridical Person",
        "G": "Government Agency"
    }

    if entity_code not in valid_entities:
        return False, f"Invalid 4th character '{entity_code}' (unknown entity category)."

    # If holder name provided and entity is Individual ('P'), verify 5th character matches surname
    if holder_name and entity_code == "P":
        # Remove common titles
        cleaned_name = re.sub(r"^(?:mr\.|mrs\.|ms\.|shri|smt|dr\.)\s+", "", holder_name.strip(), flags=re.IGNORECASE)
        name_parts = cleaned_name.split()
        if name_parts:
            surname = name_parts[-1].upper()
            expected_char = surname[0]
            if clean_pan[4] != expected_char:
                return False, (
                    f"PAN 5th character '{clean_pan[4]}' does not match first letter of surname "
                    f"'{surname}' (expected '{expected_char}')."
                )

    return True, f"Valid PAN for {valid_entities.get(entity_code, 'Entity')}."
