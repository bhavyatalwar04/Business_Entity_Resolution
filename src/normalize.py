"""Text cleaning for names and addresses. Country-agnostic, hand-written rules only."""


def clean_name(name):
    """Lowercase, strip accents, unify punctuation, expand abbreviations."""
    raise NotImplementedError


def clean_address(address):
    """Normalise address tokens (rd->road, st->street, rue, bd->boulevard, ...)."""
    raise NotImplementedError


def extract_postcode(address):
    """Pull out a postal code (6-digit PIN, 5-digit US/FR) if present."""
    raise NotImplementedError


def normalize_frame(df):
    """Raw source DataFrame -> same frame with cleaned columns added
    (name_clean, name_core, addr_clean, postcode, numbers, landmark)."""
    raise NotImplementedError
