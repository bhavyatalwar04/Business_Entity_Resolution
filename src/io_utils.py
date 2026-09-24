"""TSV reading/writing and entity-ID checks. Always read with sep='\\t'."""


def read_tsv(path):
    """Read a challenge TSV as strings, keeping empty ID lists as ''."""
    raise NotImplementedError


def write_tsv(df, path):
    """Write a DataFrame as tab-separated with no index and no quoting."""
    raise NotImplementedError


def source_of(entity_id):
    """Return 'S1', 'S2' or 'S3' from an entity_id prefix."""
    raise NotImplementedError
