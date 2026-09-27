"""France-only text canonicaliser for the cross-encoders.

Apply canon_text(name, addr, variant) to BOTH sides of every France pair before CE scoring.
India/US text is left untouched. Removes France-specific generator noise the CEs never saw in
India/US training (see the "france-perturbations" note from mine_fr.log):
  street abbreviations (R. / av / all / imp / bd ...), region vs departement tail, bracketed or
  garbled legal forms ((SARL), [SASU], 5arl, Sarl), N° / zero-padded / "37 - RUE" house numbers.

Variants (the gate decides; entry points v1 / v2 / v1keep / v2keep take and return "Name | address"):
  "v1" - accents KEPT (only the structural fixes above)
  "v2" - v1 + unidecode (also removes injected diacritics: Çlub, Àmicale, Spôrtive)
  "*keep" - same, but the region/departement tail is kept (evidence against same-name decoys)

Gate before any France rescore:
  1. ~50k fold-0 India/US rows, canonicalised with the same variant: AUC (all + contested) must not
     drop vs raw (0.9907 / 0.9025).
  2. france_catalogue.parquet: confident pairs' mean logit goes up, decoy-tagged pairs do not.

Requires: unidecode.  Self-test: python fr_canon.py
"""
import re

from unidecode import unidecode

# Address components (comma-separated) that are only a region or a departement; the city is kept.
_REGION = re.compile(
    r"^(hauts de france|nouvelle aquitaine|pays de la loire|loire atlantique|pas de calais|gironde|nord|"
    r"ile de france|occitanie|bretagne|normandie|grand est|auvergne rhone alpes|"
    r"provence alpes cote d azur|bourgogne franche comte|centre val de loire)$"
)
_STREET = {
    "r": "Rue", "av": "Avenue", "ave": "Avenue", "bd": "Boulevard", "blvd": "Boulevard",
    "all": "Allée", "imp": "Impasse", "pl": "Place", "ch": "Chemin", "rte": "Route", "crs": "Cours",
    "st": "Saint", "ste": "Sainte", "qu": "Quai", "res": "Résidence", "bat": "Bâtiment",
}
_LEGAL = {
    "5arl": "SARL", "sarl": "SARL", "sas": "SAS", "sasu": "SASU", "eurl": "EURL", "sa": "SA",
    "snc": "SNC", "sci": "SCI", "cie": "Cie",
}


def _fold(s):
    """Lower-case ASCII key, used only for matching (never for output)."""
    return unidecode(s).lower()


def _key(s):
    return re.sub(r"[^a-z]+", " ", _fold(s)).strip()


def canon_name(name, variant="v1"):
    s = name or ""
    if variant == "v2":
        s = unidecode(s)
    s = re.sub(r"[\(\)\[\]]", " ", s)  # (SARL) / [SASU] -> SARL / SASU
    s = " ".join(_LEGAL.get(_fold(t), t) for t in s.split())
    return re.sub(r"\s+", " ", s).strip()


def canon_addr(addr, variant="v1", keep_region=False):
    s = re.sub(r"\b[Nn]\s*[°º]\s*", "", addr or "")  # N°18 / Nº 18 -> 18
    if variant == "v2":
        s = unidecode(s)
    out = []
    for p in (p.strip() for p in s.split(",")):
        if not p or (not keep_region and _REGION.match(_key(p))):
            continue
        p = re.sub(r"\b0+(\d)", r"\1", p)  # 04 -> 4
        p = re.sub(r"^(\d+\w?)\s*-\s*", r"\1 ", p)  # "37 - RUE" -> "37 RUE"
        p = " ".join(_STREET.get(_fold(t).rstrip("."), t) for t in p.split())
        out.append(p)
    s = ", ".join(out)
    return unidecode(s) if variant == "v2" else s  # expansions like "Allée" must be folded too


def canon_text(name, addr, variant="v1", keep_region=False):
    """Same record format as the CE handoff: 'name | address'."""
    return f"{canon_name(name, variant)} | {canon_addr(addr, variant, keep_region)}"


def _record(text, variant, keep_region):
    name, sep, addr = (text or "").partition(" | ")
    if not sep:  # no separator: treat the whole string as a name
        return canon_name(name, variant)
    return canon_text(name, addr, variant, keep_region)


# Entry points for qwen_score.py --canon fr_canon.py:<func>; each takes and returns 'Name | address'.
def v1(text):
    return _record(text, "v1", keep_region=False)


def v2(text):
    return _record(text, "v2", keep_region=False)


def v1keep(text):
    return _record(text, "v1", keep_region=True)


def v2keep(text):
    return _record(text, "v2", keep_region=True)


if __name__ == "__main__":
    pairs = [
        (("Sauvegarde Club SASU", "37 Rue de Chartres, Tourcoing, Hauts-de-France"),
         ("Sauvegarde Club [SASU]", "TOURCOING, 37 - RUE DE CHARTRES, Nord")),
        (("Amicale Sportive Club", "12 Allée des Pins, Pessac, Nouvelle-Aquitaine"),
         ("Àmicale Spôrtive Çlub 5arl", "12 ALL DES PINS, PESSAC, Gironde")),
        (("Maison Membrane SASU", "18 Rue Augereau, Calais, Hauts-de-France"),
         ("Maison Memrane SASU", "N°18 R. Augereau, Calais, Hauts-de-France")),
        (("Nantes Compagnie SARL", "4 RUE Jeannine, chez Youssoupha Gueye, Nantes, Pays de la Loire"),
         ("Nantes  Campangie Sàrl", "04 R. Jeannine, Chez Youssoupha Gueye, Nantes, Loire-Atlantique")),
    ]
    for variant in ("v1", "v2"):
        print(f"=== {variant}")
        for s1, pool in pairs:
            print("S1  :", canon_text(*s1, variant))
            print("POOL:", canon_text(*pool, variant))
            print()
