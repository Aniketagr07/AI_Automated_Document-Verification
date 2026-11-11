"""
DocVerify - Centralized System Constants.
Defines canonical verification statuses, roles, and legal disclaimers
used across all processing, database, decision, reporting, and UI modules.
"""

# ===========================================================================
# CANONICAL VERIFICATION STATUSES
# ===========================================================================

STATUS_CONSISTENCY_PASSED = "CONSISTENCY CHECKS PASSED"
STATUS_NEEDS_REVIEW = "NEEDS MANUAL REVIEW"
STATUS_INCONSISTENCIES = "INCONSISTENCIES DETECTED"
STATUS_UNABLE_TO_VERIFY = "UNABLE TO VERIFY"

# Standard Compatibility Aliases
STATUS_PASSED = STATUS_CONSISTENCY_PASSED
STATUS_REVIEW = STATUS_NEEDS_REVIEW
STATUS_INCONSISTENT = STATUS_INCONSISTENCIES
STATUS_UNABLE = STATUS_UNABLE_TO_VERIFY

STANDARD_STATUSES = [
    STATUS_CONSISTENCY_PASSED,
    STATUS_NEEDS_REVIEW,
    STATUS_INCONSISTENCIES,
    STATUS_UNABLE_TO_VERIFY
]

# Canonical User Roles
ROLE_USER = "USER"
ROLE_ADMIN = "ADMIN"
VALID_ROLES = [ROLE_USER, ROLE_ADMIN]

# Canonical Disclaimers
AUTHENTICITY_DISCLAIMER = (
    "NOTICE: Passing automated consistency review confirms internal logical, temporal, "
    "and structural validity. It DOES NOT prove legal authenticity or absence of counterfeit origins. "
    "Legal authenticity requires issuer database queries, PKI digital signatures, or cryptographically sealed QR codes."
)
