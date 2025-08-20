import re
from datasets import load_dataset

DATASET_ID = "matheoqtb/ancre"
NEW_REPO_ID = "matheoqtb/ancre-cleaned"
TEXT_COL = "positive"

# 1) Regex combo pour retirer des lignes ENTIERES
# - >=2 occurrences de `|}`
# - lignes de tableau avec "||"
# - lignes <= 10 caractères
RE_DROP_LINES = re.compile(
    r"(?m)"
    r"^(?:[^\n]*\|\}){2,}[^\n]*$\r?\n?"         # >=2 x '|}'
    r"|^\|[^|\n]*(?:\|\|)[^|\n]*\|[^\n]*$\r?\n?" # lignes avec '||'
    r"|^(?=.{0,10}$).*\r?\n?"                    # lignes courtes (<=10)
)

def clean_text(s: str) -> str:
    if not isinstance(s, str) or not s:
        return s
    # supprime les lignes ciblées
    s = RE_DROP_LINES.sub("", s)
    # trim des espaces de fin de ligne + compactage des vides
    s = re.sub(r"[ \t]+\r?$", "", s, flags=re.M)
    s = re.sub(r"\n{2,}", "\n", s).strip()
    return s

# Charge
ds = load_dataset(DATASET_ID)

# Nettoie ligne par ligne dans la colonne 'positive'
ds = ds.map(lambda ex: {TEXT_COL: clean_text(ex[TEXT_COL])})

# 2) En plus, on enlève toute entrée dont 'positive' fait <= 10 caractères au final
ds = ds.filter(lambda ex: isinstance(ex[TEXT_COL], str) and len(ex[TEXT_COL]) > 10)

# Push sur le Hub
ds.push_to_hub(NEW_REPO_ID)
