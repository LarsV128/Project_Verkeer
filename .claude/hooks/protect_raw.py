"""PreToolUse-hook: blokkeert elke wijziging aan bestanden in de map data/raw/.

Claude Code stuurt een JSON-bericht via stdin. Antwoord met een 'deny'-beslissing
als de tool in data/raw/ wil schrijven; geen uitvoer + exit 0 betekent 'toegelaten'.
"""
import json
import os
import re
import sys

MELDING = (
    "Geblokkeerd: de map 'data/raw' is alleen-lezen. Wijzig nooit bestanden in data/raw/. "
    "Lees de ruwe data en sla de opgeschoonde versie op in de map 'data/' "
    "met een duidelijke naam, bv. data/OPENDATA_MAP_2017-2024_clean.csv."
)

# Bash/PowerShell: een commando dat data/raw/ noemt EN iets schrijft of verwijdert.
RAW_PAD = r"""data[/\\]raw[/\\]"""
SCHRIJF_COMMANDO = re.compile(
    r"\b(rm|del|erase|rmdir|mv|move|ren|rename|Remove-Item|Move-Item|Rename-Item|"
    r"Set-Content|Add-Content|Clear-Content|Out-File|New-Item|sed\s+-i|truncate)\b",
    re.IGNORECASE,
)
OMLEIDING_NAAR_RAW = re.compile(r">>?\s*['\"]?(?:\.[/\\])?data[/\\]raw[/\\]", re.IGNORECASE)

# Python-code in een commando. We bekijken per functieaanroep de argumenten tussen haakjes.
PYTHON_AANROEP = re.compile(r"\b(\w+)\s*\(([^()]*)\)")
# Altijd schrijven/verwijderen: een raw-pad in de argumenten is genoeg om te blokkeren.
PYTHON_SCHRIJVERS = {
    "to_csv", "to_excel", "to_parquet", "to_json", "to_pickle", "to_feather", "write_text", "write_bytes",
    "savefig", "remove", "unlink", "rmtree", "rename", "replace", "move",
}
PYTHON_KOPIE = {"copy", "copy2", "copyfile", "copytree"}  # lezen uit raw mag; enkel de bestemming telt
SCHRIJFMODUS = re.compile(r"""['"][^'"]*[wax+][^'"]*['"]""", re.IGNORECASE)
# Path("data/raw/x").unlink() e.d.: de methode staat NA het raw-pad
RAW_PAD_METHODE = re.compile(
    r"""data[/\\]raw[/\\][^'"]*['"]\s*\)\s*\.\s*(unlink|write_text|write_bytes|rename|replace|touch|rmdir|open)\s*\(""",
    re.IGNORECASE,
)


def python_schrijft_in_raw(commando):
    """True als python-code in het commando iets naar data/raw/ schrijft, kopieert of verwijdert."""
    for m in PYTHON_AANROEP.finditer(commando):
        naam, args = m.group(1).lower(), m.group(2)
        if not re.search(RAW_PAD, args, re.IGNORECASE):
            continue
        if naam in PYTHON_SCHRIJVERS:
            return True
        if naam == "open":  # open(pad) of open(pad, "r") leest enkel; kijk dus niet naar het pad zelf
            delen = args.split(",")
            modus = [d for d in delen[1:2] if "=" not in d] + [d for d in delen[1:] if re.match(r"\s*mode\s*=", d)]
            if any(SCHRIJFMODUS.search(d) for d in modus):
                return True
        if naam in PYTHON_KOPIE:
            delen = args.split(",", 1)
            if len(delen) == 2 and re.search(RAW_PAD, delen[1], re.IGNORECASE):
                return True
    meth = RAW_PAD_METHODE.search(commando)
    return bool(meth and meth.group(1).lower() != "open")


def in_raw(pad, project_dir):
    """True als `pad` binnen <project>/data/raw valt."""
    if not pad:
        return False
    if not os.path.isabs(pad):
        pad = os.path.join(project_dir, pad)
    pad = os.path.normcase(os.path.realpath(pad))
    raw = os.path.normcase(os.path.realpath(os.path.join(project_dir, "data", "raw")))
    return pad == raw or pad.startswith(raw + os.sep)


def bash_schrijft_in_raw(commando):
    if not re.search(RAW_PAD, commando, re.IGNORECASE):
        return False
    return bool(
        SCHRIJF_COMMANDO.search(commando)
        or OMLEIDING_NAAR_RAW.search(commando)
        or python_schrijft_in_raw(commando)
    )


def main():
    try:
        data = json.load(sys.stdin)
    except ValueError:
        return  # onleesbaar bericht: niets blokkeren, ook niet crashen
    project_dir = os.environ.get("CLAUDE_PROJECT_DIR") or data.get("cwd") or os.getcwd()
    tool = data.get("tool_name", "")
    inp = data.get("tool_input", {}) or {}

    if tool in ("Write", "Edit", "NotebookEdit"):
        geblokkeerd = in_raw(inp.get("file_path") or inp.get("notebook_path"), project_dir)
    elif tool in ("Bash", "PowerShell"):
        geblokkeerd = bash_schrijft_in_raw(inp.get("command", ""))
    else:
        geblokkeerd = False

    if geblokkeerd:
        print(json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": MELDING,
            }
        }))


if __name__ == "__main__":
    main()
