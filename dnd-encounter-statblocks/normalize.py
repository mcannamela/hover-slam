"""Turn raw monster-service JSON into display-ready fields, matching the
2024 D&D Beyond stat-block layout (e.g. dndbeyond.com/monsters/5194864-aarakocra-aeromancer).
"""
import json
import math
import re
from bs4 import BeautifulSoup

STAT_NAMES = {1: "STR", 2: "DEX", 3: "CON", 4: "INT", 5: "WIS", 6: "CHA"}
ADJ_TYPE_LABEL = {1: "Resistances", 2: "Immunities", 3: "Vulnerabilities"}

_AXIS = {"Lawful": "L", "Chaotic": "C", "Good": "G", "Evil": "E", "Neutral": "N"}


def abbreviate_alignment(name):
    if name in ("Unaligned",):
        return name
    prefix = ""
    rest = name
    if rest.startswith("Typically "):
        prefix, rest = "T. ", rest[len("Typically "):]
    if rest == "Any Alignment":
        return prefix + "Any"
    m = re.match(r"^Any (Non-)?(\w+) Alignment$", rest)
    if m:
        neg, axis = m.groups()
        return prefix + "Any " + ("Non-" if neg else "") + _AXIS.get(axis, axis)
    words = rest.split()
    if all(w in _AXIS for w in words):
        return prefix + "".join(_AXIS[w] for w in words)
    return prefix + rest


NUM_WORDS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}
_ABILITY_ABBR = {
    "strength": "STR", "dexterity": "DEX", "constitution": "CON",
    "intelligence": "INT", "wisdom": "WIS", "charisma": "CHA",
}


def _placeholder_removed_links(html):
    """D&D Beyond's homebrew data sometimes has dead cross-references baked in as
    literal '<span class="link-removed">Link Removed: <url></span>' text (the source
    page for that condition was retired). Stash them as plain-text placeholders
    (immune to tag-stripping regexes below) to be restored as real links afterward."""
    def repl(m):
        url = m.group(1)
        frag = url.split("#")[-1] if "#" in url else url.rstrip("/").rsplit("/", 1)[-1]
        label = re.sub(r"(?<!^)(?=[A-Z])", " ", frag).replace("-", " ").strip() or "link"
        return f'@@LINK@@{url}@@{label}@@'
    return re.sub(
        r'<span class="link-removed">\s*Link Removed:\s*(\S+?)\s*</span>',
        repl, html, flags=re.I,
    )


def _restore_removed_links(html):
    return re.sub(
        r'@@LINK@@(.*?)@@(.*?)@@',
        r'<a class="xref" href="\1">\2</a>',
        html,
    )


def tersify(html):
    """Fixed, mechanical shortenings to squeeze more text into less space.
    No rewriting/summarizing -- just consistent word/phrase substitutions."""
    html = re.sub(r'\bMelee or Ranged Weapon Attack\b', 'Melee/Ranged', html, flags=re.I)
    html = re.sub(r'\bMelee or Ranged Attack Roll\b', 'Melee/Ranged', html, flags=re.I)
    html = re.sub(r'\bMelee Weapon Attack\b', 'Melee', html, flags=re.I)
    html = re.sub(r'\bRanged Weapon Attack\b', 'Ranged', html, flags=re.I)
    html = re.sub(r'\bMelee Attack Roll\b', 'Melee', html, flags=re.I)
    html = re.sub(r'\bRanged Attack Roll\b', 'Ranged', html, flags=re.I)
    html = re.sub(r'\bMelee Spell Attack\b', 'Melee (spell)', html, flags=re.I)
    html = re.sub(r'\bRanged Spell Attack\b', 'Ranged (spell)', html, flags=re.I)

    def save_repl(m):
        return f"{_ABILITY_ABBR[m.group(1).lower()]} save"
    html = re.sub(r'\b(Strength|Dexterity|Constitution|Intelligence|Wisdom|Charisma)\s+[Ss]aving [Tt]hrow\b',
                  save_repl, html)
    html = re.sub(r'\b[Ss]aving [Tt]hrow\b', 'save', html)

    html = re.sub(r'\battacks\b', 'atk', html, flags=re.I)
    html = re.sub(r'\battack\b', 'atk', html, flags=re.I)

    def num_repl(m):
        return NUM_WORDS[m.group(0).lower()]
    html = re.sub(r'\b(' + '|'.join(NUM_WORDS) + r')\b', num_repl, html, flags=re.I)

    return html


def load_config():
    with open("config.json") as f:
        cfg = json.load(f)
    idx = {}
    for key in ("monsterTypes", "challengeRatings", "alignments", "senses",
                "creatureSizes", "movements", "languages", "damageAdjustments",
                "abilitySkills"):
        idx[key] = {row["id"]: row for row in cfg[key]}
    return idx


def mod(score):
    return math.floor((score - 10) / 2)


def fmt_signed(n):
    return f"+{n}" if n >= 0 else str(n)


def _absolutize_links(html):
    """D&D Beyond's tooltip cross-reference links (conditions, spells, skills,
    other monsters, etc.) use site-relative hrefs. Keep them as real clickable
    links instead of stripping to plain underlined text."""
    def repl(m):
        href, text = m.group(1), m.group(2)
        if href.startswith("/"):
            href = "https://www.dndbeyond.com" + href
        return f'<a class="xref" href="{href}">{text}</a>'
    return re.sub(r'<a\b[^>]*\bhref="([^"]*)"[^>]*>(.*?)</a>', repl, html, flags=re.S)


def strip_html_xrefs(html):
    """Turn D&D Beyond's rich-text HTML into our simplified markup: keep tooltip
    <a> links as real clickable links, unwrap dice-notation <span> wrappers."""
    if not html:
        return ""
    html = _placeholder_removed_links(html)
    html = _absolutize_links(html)
    html = re.sub(r'<span\b[^>]*>(.*?)</span>', r'\1', html, flags=re.S)
    html = _restore_removed_links(html)
    return html


def parse_entries(html):
    """Split a *Description HTML blob into a list of {name, body_html} entries,
    one per top-level <p> that starts with an <em><strong>Name.</strong></em> lead-in.
    Falls back to treating the whole blob as one unnamed entry."""
    if not html:
        return []
    html = strip_html_xrefs(html)
    parts = re.findall(r'<p>(.*?)</p>', html, flags=re.S)
    entries = []
    for p in parts:
        frag = BeautifulSoup(p, "html.parser")
        strong = frag.find("strong")
        if strong and not strong.find_previous(string=re.compile(r"\S")):
            name = re.sub(r"\.\s*$", "", strong.get_text()).strip()
            strong.decompose()
            body = str(frag).strip()
            body = re.sub(r"^(<em>\s*</em>)+", "", body).strip()
        else:
            name = None
            body = p.strip()
        entries.append({"name": name, "body": tersify(body)})
    return entries


def format_cr(cr_entry):
    v = cr_entry["value"]
    if v == 0:
        s = "0"
    elif v == int(v):
        s = str(int(v))
    else:
        frac = {0.125: "1/8", 0.25: "1/4", 0.5: "1/2"}.get(v, str(v))
        s = frac
    return s, cr_entry["xp"], cr_entry["proficiencyBonus"]


def normalize(monster, idx):
    stats = {s["statId"]: s["value"] for s in monster["stats"]}
    prof_bonus = 2
    cr_str, cr_xp, prof_bonus = format_cr(idx["challengeRatings"][monster["challengeRatingId"]])

    save_profs = {}
    for st in monster["savingThrows"]:
        base = mod(stats[st["statId"]])
        save_profs[st["statId"]] = st["bonusModifier"] if st["bonusModifier"] is not None else base + prof_bonus

    ability_rows = []
    for sid in (1, 2, 3, 4, 5, 6):
        score = stats[sid]
        m = mod(score)
        save = save_profs.get(sid, m)
        ability_rows.append({
            "key": STAT_NAMES[sid], "score": score,
            "mod": fmt_signed(m), "save": fmt_signed(save),
        })

    # speed
    speed_parts = []
    walk = next((mv for mv in monster["movements"] if mv["movementId"] == 1), None)
    if walk:
        speed_parts.append(f"{walk['speed']} ft.")
    for mv in monster["movements"]:
        if mv["movementId"] == 1:
            continue
        name = idx["movements"][mv["movementId"]]["name"]
        speed_parts.append(f"{name} {mv['speed']} ft.")
    speed = ", ".join(speed_parts) if speed_parts else "0 ft."

    # initiative: no explicit override in our data -> default to DEX mod
    init_bonus = monster.get("initiativeBonus")
    if init_bonus is None:
        init_bonus = mod(stats[2])
    init_passive = 10 + init_bonus

    size = idx["creatureSizes"].get(monster["sizeId"], {}).get("name", "")
    ctype = idx["monsterTypes"].get(monster["typeId"], {}).get("name", "")
    alignment = idx["alignments"].get(monster["alignmentId"], {}).get("name", "Unaligned")
    alignment_abbr = abbreviate_alignment(alignment)

    skills_html = strip_html_xrefs(monster.get("skillsHtml") or "")
    senses_html = strip_html_xrefs(monster.get("sensesHtml") or "")
    senses = f"{senses_html}, Passive Perception {monster['passivePerception']}" if senses_html else f"Passive Perception {monster['passivePerception']}"

    langs = []
    for lg in monster.get("languages", []):
        name = idx["languages"].get(lg["languageId"], {}).get("name", "?")
        note = f" {lg['notes']}" if lg.get("notes") else ""
        langs.append(name + note)
    if monster.get("languageDescription"):
        langs.append(monster["languageDescription"])
    languages = ", ".join(langs) if langs else "—"
    if monster.get("languageNote"):
        languages += f" ({monster['languageNote']})"

    cond_imm = strip_html_xrefs(monster.get("conditionImmunitiesHtml") or "")

    dmg_groups = {1: [], 2: [], 3: []}
    for adj_id in monster.get("damageAdjustments", []):
        row = idx["damageAdjustments"].get(adj_id)
        if row:
            dmg_groups[row["type"]].append(row["name"])
    damage_lines = []
    for t in (1, 2, 3):
        if dmg_groups[t]:
            damage_lines.append((ADJ_TYPE_LABEL[t], ", ".join(dmg_groups[t])))

    traits = parse_entries(monster.get("specialTraitsDescription"))
    actions = parse_entries(monster.get("actionsDescription"))
    bonus_actions = parse_entries(monster.get("bonusActionsDescription"))
    reactions = parse_entries(monster.get("reactionsDescription"))
    legendary = parse_entries(monster.get("legendaryActionsDescription"))
    mythic = parse_entries(monster.get("mythicActionsDescription"))

    return {
        "name": monster["name"],
        "count": monster.get("count", 1),
        "subtitle": f"{size} {ctype}, {alignment_abbr}",
        "ac": monster["armorClass"],
        "ac_desc": monster.get("armorClassDescription") or "",
        "hp": monster["averageHitPoints"],
        "hp_dice": monster["hitPointDice"]["diceString"],
        "speed": speed,
        "init": fmt_signed(init_bonus),
        "init_passive": init_passive,
        "ability_rows": ability_rows,
        "skills": skills_html,
        "senses": senses,
        "condition_immunities": cond_imm,
        "damage_lines": damage_lines,
        "languages": languages,
        "cr": cr_str,
        "prof_bonus": fmt_signed(prof_bonus),
        "traits": traits,
        "actions": actions,
        "bonus_actions": bonus_actions,
        "reactions": reactions,
        "legendary": legendary,
        "mythic": mythic,
        "avatar_url": monster.get("avatarUrl"),
    }
