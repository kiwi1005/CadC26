"""Structured corruption generators for repair learning."""

from hcfp.repair.corruption.contact import (
    ContactCorruption,
    contact_c2_eligible,
    generate_contact_corruptions,
)


__all__ = ["ContactCorruption", "contact_c2_eligible", "generate_contact_corruptions"]
