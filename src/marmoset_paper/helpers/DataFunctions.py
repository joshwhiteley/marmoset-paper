"""Identifier normalization shared by analysis boundaries."""


def normalize_regimen(name: str) -> str:
    """Map the historical OPC label to QBS without changing regimen composition.

    Apply aliases to whole drug tokens, not substrings. Raw input files retain
    their deposited labels. Drug order is deliberately left unchanged.
    """
    aliases = {"OPC": "QBS"}
    return "+".join(
        aliases.get(drug.strip().upper(), drug.strip().upper()) for drug in name.split("+")
    )
