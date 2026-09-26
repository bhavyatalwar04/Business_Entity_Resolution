"""Synthetic labelled pairs for an unseen country, built by re-applying the dataset generator's perturbations.

The train generator turns one Source-1 record into Source-2/3 copies with surface noise (case, OCR typos, injected
accents, noise prefixes, brackets, legal-form swaps, dba wrappers, handle/domain forms, address field shuffles and drops,
street-type abbreviations, region/department swaps, house-number prefixes) and plants hard negatives (decoys): the same or
a near-same name with a nearby house number, the name plus an inserted industry word, or a different business at the
exact same address. Applying the same operators to test Source-1 records of an unseen country gives pairs with KNOWN
labels in that country's surface forms, which the cross-encoder can be fine-tuned on without any external data.

Positive / decoy rates follow the train statistics measured on 100k gold S1 (see logs/synth_stats.txt):
gold pairs keep a shared house number 91.5% of the time; decoys have an empty address 45% of the time.

Usage (library): build_pairs(anchors, country, rng, n_pos=2, n_dec=2) -> DataFrame(s1_id, pool_id, label, pool_text, kind)
"""
import re
import unicodedata

import numpy as np
import pandas as pd

LEXICON = {
    "FR": {
        "legal": ["SARL", "SAS", "EURL", "SA", "SASU", "SCI", "Sàrl", "S.A.S", "S.A.R.L.", "SNC"],
        "street": {"RUE": ["R", "R."], "AVENUE": ["AV", "AV.", "AVE"], "BOULEVARD": ["BD", "BLD", "BVD"],
                   "PLACE": ["PL", "PL."], "CHEMIN": ["CH", "CHE"], "ALLEE": ["ALL", "ALL."], "ALLÉE": ["ALL", "ALLEE"],
                   "IMPASSE": ["IMP"], "ROUTE": ["RTE"], "QUAI": ["QU"], "COURS": ["CRS"], "SQUARE": ["SQ"]},
        "num_prefix": ["N°", "N° ", "No ", "n°"],
        "num_suffix": ["BIS", " BIS", "TER", " bis"],
        "insert": ["Services", "Conseil", "Distribution", "Holding", "Participations", "Groupe", "International",
                   "Immobilier", "Pharmacie", "Transports", "Club", "Centre", "Associés", "& Fils", "& Cie", "(France)"],
        "dba": ["{b} exerçant sous {a}", "{a} (enseigne {b})"],
    },
    "US": {
        "legal": ["LLC", "Inc", "Corp", "Co", "L.L.C.", "Corporation", "Incorporated", "Ltd"],
        "street": {"STREET": ["ST", "ST."], "AVENUE": ["AVE", "AV"], "BOULEVARD": ["BLVD"], "DRIVE": ["DR"],
                   "ROAD": ["RD"], "COURT": ["CT"], "LANE": ["LN"], "PLACE": ["PL"], "HIGHWAY": ["HWY"],
                   "PARKWAY": ["PKWY"], "CIRCLE": ["CIR"], "TERRACE": ["TER"]},
        "num_prefix": ["#", "##", "No. "],
        "num_suffix": ["A", "B", "C", "-"],
        "insert": ["Services", "Group", "Holdings", "Solutions", "Partners", "Clinic", "Consulting", "Medical",
                   "Enterprises", "International", "Supply", "Associates"],
        "dba": ["{b} dba {a}", "{b} d/b/a {a}", "{b} DBA {a}"],
    },
}
NOISE_PREFIX = ["--", "...", ">>", "***", "#", "- ", "** "]
OCR = {"o": "0", "O": "0", "l": "1", "i": "1", "I": "1", "s": "5", "S": "5", "e": "3", "a": "@", "B": "8", "g": "9"}
ACCENT = {"a": "àâá", "e": "éèêë", "i": "îï", "o": "ôó", "u": "ùûü", "c": "ç", "A": "ÀÁ", "E": "ÉÈ", "I": "Í", "O": "Ó"}
NUM_RE = re.compile(r"\b(\d+)([A-Za-z]{0,3})\b")


def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def case(s, rng):
    r = rng.rand()
    return s.upper() if r < 0.5 else s.lower() if r < 0.65 else s.title() if r < 0.8 else s


def typo(s, rng):
    if len(s) < 4:
        return s
    i = rng.randint(1, len(s) - 1)
    r = rng.rand()
    c = s[i]
    if r < 0.35 and c in OCR:
        return s[:i] + OCR[c] + s[i + 1:]
    if r < 0.6:
        return s[:i] + s[i + 1:]
    if r < 0.8:
        return s[:i] + c + s[i:]
    return s[:i] + s[i + 1] + s[i] + s[i + 2:] if i + 2 <= len(s) else s


def accent(s, rng):
    idx = [i for i, c in enumerate(s) if c in ACCENT]
    if not idx:
        return s
    i = idx[rng.randint(len(idx))]
    opts = ACCENT[s[i]]
    return s[:i] + opts[rng.randint(len(opts))] + s[i + 1:]


def legal_swap(name, lex, rng):
    words = name.split()
    forms = {f.upper().strip(".") for f in lex["legal"]}
    keep = [w for w in words if w.upper().strip(".()") not in forms]
    r = rng.rand()
    if r < 0.35 or len(keep) == len(words) and r < 0.6:
        keep.append(lex["legal"][rng.randint(len(lex["legal"]))])
    elif r < 0.5 and keep:
        keep.insert(0, lex["legal"][rng.randint(len(lex["legal"]))])
    return " ".join(keep) if keep else name


def name_ops(name, lex, rng, k):
    ops = ["case", "typo", "accent", "noise", "bracket", "legal", "reorder", "strip_acc", "dba", "handle"]
    p = np.array([3, 3, 2, 1, 1, 3, 1, 1, 0.4, 0.3])
    for op in rng.choice(ops, size=k, replace=False, p=p / p.sum()):
        if op == "case":
            name = case(name, rng)
        elif op == "typo":
            name = typo(name, rng)
        elif op == "accent":
            name = accent(name, rng)
        elif op == "strip_acc":
            name = strip_accents(name)
        elif op == "noise":
            name = NOISE_PREFIX[rng.randint(len(NOISE_PREFIX))] + " " + name
        elif op == "bracket":
            w = name.split()
            if len(w) > 1:
                j = rng.randint(len(w))
                w[j] = ("[%s]" if rng.rand() < 0.5 else "(%s)") % w[j]
                name = " ".join(w)
        elif op == "legal":
            name = legal_swap(name, lex, rng)
        elif op == "reorder":
            w = name.split()
            if len(w) > 2:
                j = rng.randint(1, len(w))
                name = " ".join(w[j:] + w[:j])
        elif op == "dba":
            tmpl = lex["dba"][rng.randint(len(lex["dba"]))]
            name = tmpl.format(a=name, b=random_brand(rng))
        elif op == "handle":
            core = re.sub(r"[^a-z0-9]", "", strip_accents(name).lower())[:18]
            name = ("#" + core) if rng.rand() < 0.5 else core + (".fr" if lex is LEXICON["FR"] else ".com")
    return name.strip()


def random_brand(rng):
    syl = ["ly", "ra", "zo", "nex", "ph", "sol", "gild", "kor", "vex", "ta", "mi", "on", "del", "qua", "ri"]
    return "".join(syl[rng.randint(len(syl))] for _ in range(rng.randint(2, 4))).title()


def abbreviate(part, lex, rng):
    words = part.split()
    for j, w in enumerate(words):
        key = strip_accents(w.upper().strip("."))
        for full, abbr in lex["street"].items():
            if key == strip_accents(full):
                words[j] = abbr[rng.randint(len(abbr))] if rng.rand() < 0.7 else full
                break
            if key in {strip_accents(a).strip(".") for a in abbr}:
                words[j] = full if rng.rand() < 0.6 else w
                break
    return " ".join(words)


def number_prefix(part, lex, rng):
    m = NUM_RE.search(part)
    if not m:
        return part
    pre = lex["num_prefix"][rng.randint(len(lex["num_prefix"]))]
    return part[:m.start()] + pre + part[m.start():]


def shift_number(addr, lex, rng):
    """Decoy house number: a nearby but different number (or a BIS/letter suffix change)."""
    m = NUM_RE.search(addr)
    if not m:
        return None
    n, suf = int(m.group(1)), m.group(2)
    r = rng.rand()
    if r < 0.2 and not suf:
        new = f"{n}{lex['num_suffix'][rng.randint(len(lex['num_suffix']))]}"
    else:
        d = int(rng.choice([1, 2, 3, 4, 6, 10, 12, 20])) * (1 if rng.rand() < 0.5 else -1)
        new = str(max(1, n + d)) if max(1, n + d) != n else str(n + 1)
    return addr[:m.start()] + new + addr[m.end():]


def addr_ops(addr, lex, rng, k):
    parts = [p.strip() for p in addr.split(",") if p.strip()]
    ops = ["case", "abbr", "shuffle", "drop", "numpre", "typo", "strip_acc"]
    p = np.array([3, 3, 2, 1.5, 1, 1.5, 1])
    for op in rng.choice(ops, size=min(k, len(ops)), replace=False, p=p / p.sum()):
        if op == "case":
            parts = [case(x, rng) if rng.rand() < 0.8 else x for x in parts]
        elif op == "abbr" and parts:
            parts[0] = abbreviate(parts[0], lex, rng)
        elif op == "shuffle" and len(parts) > 1:
            parts = list(np.array(parts, dtype=object)[rng.permutation(len(parts))])
        elif op == "drop" and len(parts) > 2:
            del parts[rng.randint(1, len(parts))]
        elif op == "numpre" and parts:
            parts[0] = number_prefix(parts[0], lex, rng)
        elif op == "typo" and parts:
            j = rng.randint(len(parts))
            parts[j] = typo(parts[j], rng)
        elif op == "strip_acc":
            parts = [strip_accents(x) for x in parts]
    return ", ".join(parts)


REGIONS_FR = {"hauts-de-france", "nouvelle-aquitaine", "pays de la loire", "nord", "gironde", "pas-de-calais",
              "loire-atlantique", "ile-de-france", "île-de-france"}


def drop_context(addr, rng):
    """France hard positives (laptop #61): the copy lost its region/departement tail and sometimes the street."""
    parts = [p.strip() for p in addr.split(",") if p.strip()]
    parts = [p for p in parts if strip_accents(p).lower() not in {strip_accents(x) for x in REGIONS_FR}] or parts
    if len(parts) > 1 and rng.rand() < 0.3:
        m = NUM_RE.search(parts[0])
        parts[0] = parts[0][m.start():m.end()] if m and rng.rand() < 0.5 else parts[0].split()[-1]
    return ", ".join(parts)


def positive(name, addr, lex, rng):
    n = name_ops(name, lex, rng, k=int(rng.choice([1, 2, 3], p=[0.4, 0.4, 0.2])))
    r = rng.rand()
    if lex is LEXICON["FR"] and r < 0.45:
        addr = drop_context(addr, rng)
    a = "" if r > 0.96 else addr_ops(addr, lex, rng, k=int(rng.choice([1, 2, 3], p=[0.35, 0.45, 0.2])))
    return f"{n} | {a}"


def decoy(name, addr, lex, rng, other_names):
    """Hard negative in the generator's decoy styles. Returns (text, kind) or (None, None)."""
    r = rng.rand()
    if r < 0.40:  # same entity name, nearby different house number
        a2 = shift_number(addr, lex, rng)
        if a2 is None:
            return None, None
        n = name_ops(name, lex, rng, k=1) if rng.rand() < 0.6 else name
        return f"{n} | {addr_ops(a2, lex, rng, k=1)}", "num_shift"
    if r < 0.65:  # name + inserted industry word, same or nearby address
        w = name.split()
        j = rng.randint(1, len(w) + 1) if len(w) else 0
        ins = lex["insert"][rng.randint(len(lex["insert"]))]
        n = " ".join(w[:j] + [ins] + w[j:])
        a2 = addr if rng.rand() < 0.5 else (shift_number(addr, lex, rng) or addr)
        return f"{name_ops(n, lex, rng, k=1)} | {addr_ops(a2, lex, rng, k=1)}", "insert_word"
    # different real business name at the exact same address
    other = other_names[rng.randint(len(other_names))]
    if strip_accents(other).lower() == strip_accents(name).lower():
        return None, None
    return f"{name_ops(other, lex, rng, k=1)} | {addr_ops(addr, lex, rng, k=1)}", "same_addr"


def build_pairs(anchors, country, rng, n_pos=2, n_dec=2, id_prefix="SYN"):
    """anchors: DataFrame(entity_id, business_name, business_address) of ONE country. Returns labelled synthetic pairs."""
    lex = LEXICON[country]
    other = anchors["business_name"].to_numpy()
    rows = []
    for eid, name, addr in zip(anchors["entity_id"], anchors["business_name"], anchors["business_address"]):
        for _ in range(n_pos):
            rows.append((eid, positive(name, addr, lex, rng), 1, "pos"))
        for _ in range(n_dec):
            t, kind = decoy(name, addr, lex, rng, other)
            if t is not None:
                rows.append((eid, t, 0, kind))
    df = pd.DataFrame(rows, columns=["s1_id", "pool_text", "label", "kind"])
    df.insert(1, "pool_id", [f"{id_prefix}-{country}-{i}" for i in range(len(df))])
    return df
