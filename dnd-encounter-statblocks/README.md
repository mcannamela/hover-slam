# D&D Beyond Encounter Stat Blocks

Pull every monster's full stat block out of a D&D Beyond encounter and render
them as a compact, printable PDF (2024-monster-format style, two columns per
landscape page). Built to prep for in-person D&D sessions without having to
flip between browser tabs mid-combat.

This works for **any** D&D Beyond encounter you have view access to — just
swap in the encounter URL/id. There's nothing "Faerie Rescue"-specific baked
in; that was just the first encounter it was built and tested against.

## How it works

D&D Beyond is a client-rendered SPA — the page HTML has none of the actual
monster data. Instead:

1. `login.py` drives a **real installed Chrome** (not Playwright's bundled
   Chromium) with a dedicated fresh profile, because Google's Sign-In bot
   detection ("This browser or app may not be secure") flags the bundled
   Chromium's automation fingerprint. It waits for a real `CobaltSession`
   cookie to appear (not just any redirect away from `/login`, which can
   happen mid-flow) and saves the session to `auth.json`.
2. `fetch_stat_blocks.py` replays that session with headless Playwright,
   loads the encounter page, and listens for two XHR calls the page itself
   makes:
   - `encounter-service.dndbeyond.com/v1/encounters/<id>` — the monster
     list + per-instance HP/quantities for the encounter.
   - `monster-service.dndbeyond.com/v1/Monster?ids=...` — the actual full
     stat blocks (stats, AC, actions, traits, etc.) for the unique monster
     ids in that encounter. This call is normally only fired when you click
     a monster or hit "Run Encounter" in the UI, so the script triggers it
     directly via `page.request.get(...)` if it doesn't see it fire on load.
   It also fetches `www.dndbeyond.com/api/config/json`, a **public,
   unauthenticated** endpoint containing every lookup table the monster data
   references by id: ability names, creature sizes/types, alignments,
   challenge ratings (value + XP + proficiency bonus), movement types,
   senses, languages, damage resistance/immunity/vulnerability types, skills.
3. `normalize.py` decodes the raw ids into display values (ability
   modifiers/saves, CR, speed string, etc.), rewrites D&D Beyond's rich-text
   HTML into something safe to drop into our own template, and applies fixed
   text shortenings (see below).
4. `template.html` + `build_pdf.py` render everything through Jinja2 into an
   HTML page styled like the modern D&D Beyond stat-block layout, then use
   Playwright's `page.pdf()` (a real headless-Chrome print) to produce the
   final PDF. Cards are packed via real CSS multi-column (`column-count: 2`)
   so each column fills independently — short and long stat blocks pack
   tightly instead of being stretched to match whatever's next to them.

## Setup

```bash
cd dnd-encounter-statblocks
uv sync
uv run playwright install chromium   # only needed for the headless fetch step
```

Real Chrome (`google-chrome` / `google-chrome-stable`) must already be
installed system-wide — that's what `login.py` drives for the login step.

## Usage

```bash
# One-time (or whenever the session expires): log in interactively.
# Opens a real Chrome window -- log in there, script auto-detects success.
uv run python login.py

# Fetch an encounter's monsters + stat blocks.
uv run python fetch_stat_blocks.py https://www.dndbeyond.com/encounters/<uuid>
# (a bare uuid also works)

# Render the PDF.
uv run python build_pdf.py [output.pdf]   # defaults to stat_blocks.pdf
```

All commands assume your working directory is this project folder (they
read/write `auth.json`, `config.json`, `monsters_full.json`, etc. relative to
cwd, not to the script's location).

## Session expiry

`auth.json` holds real login cookies (`CobaltSession` etc.) and is
gitignored — never commit it. If `fetch_stat_blocks.py` starts failing (302
redirect to the campaigns homepage instead of the encounter), the session's
expired; just rerun `login.py`.

## Notable gotchas hit building this (useful if extending it)

- **Google Sign-In bot detection**: don't use Playwright's bundled Chromium
  for the interactive login — use `channel="chrome"` against a real Chrome
  install with a fresh `launch_persistent_context` profile dir, plus
  `--disable-blink-features=AutomationControlled`. Even then, wait for the
  actual session cookie, not just a URL change — the login flow redirects
  through several intermediate URLs before it's really done.
- **CSS multi-column + `break-inside: avoid` "ghost overlap" bug**: what
  looked like a Chromium print-pagination bug (duplicated/overlapping text
  at the top of a card) was actually the ability-score sub-table overflowing
  its containing cell because the left column was too narrow for `nowrap`
  text like "STR 16" at small font sizes. Widening that column (and/or
  reducing font size) fixed it outright — the multi-column layout itself is
  fine and packs correctly (confirmed after the real fix landed).
- **D&D Beyond's homebrew data has some dead cross-references**: some
  monster text has literal `<span class="link-removed">Link Removed:
  https://...#ConditionName</span>` baked in where a linked page was
  retired. We turn these into a real link using the URL fragment
  (`#Shocked` -> "Shocked") as the visible label, since the underlying URL
  still resolves fine.
- **Tag order varies**: some monster entries use `<em><strong>Name.</strong>
  ...</em>` and others `<strong><em>Name.</em></strong>` or a bare
  `<strong>Name.</strong>` — `normalize.parse_entries` uses BeautifulSoup
  (not a fixed regex) to pull out the first `<strong>` at the very start of
  a paragraph as the entry name, specifically so it doesn't care which
  order/nesting the source used.
- Ability score IDs are always `1..6 = STR, DEX, CON, INT, WIS, CHA` in that
  order — see `STAT_NAMES` in `normalize.py`.
- `damageAdjustments` type codes (from `config.json`): `1 = resistance,
  2 = immunity, 3 = vulnerability`.

## Text shortenings applied ("tersify" in `normalize.py`)

Fixed, mechanical substitutions only — no paraphrasing/rewriting, to avoid
silently changing rules text:

- `Melee/Ranged Weapon Attack` / `... Attack Roll` -> `Melee` / `Ranged`
  (2014- and 2024-phrasing both handled)
- `<Ability> Saving Throw` -> `<ABBR> save` (e.g. `Dexterity Saving Throw` ->
  `DEX save`); bare `saving throw` -> `save`
- `attack(s)` -> `atk`
- number words `one`..`ten` -> digits

## Possible future improvements

- Multiple encounters in one PDF (currently one encounter per run/file).
- A CLI flag for portrait vs. landscape / columns-per-page instead of the
  hardcoded landscape 2-up layout in `template.html`.
- Legendary/lair actions styling for monsters with lair descriptions
  (`lairDescription` is fetched but not currently rendered).
