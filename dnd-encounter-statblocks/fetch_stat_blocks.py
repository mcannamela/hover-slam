"""Fetch an encounter's full monster stat blocks from D&D Beyond.

Usage: uv run python fetch_stat_blocks.py <encounter_url_or_id>

Produces monsters_full.json: {
  "encounter_name": str,
  "monsters": [ {..raw monster-service fields.., "count": int}, ... ]
}
and config.json (lookup tables, public endpoint, no auth needed).
"""
import json
import re
import sys
import urllib.request
from playwright.sync_api import sync_playwright

AUTH_FILE = "auth.json"
CONFIG_URL = "https://www.dndbeyond.com/api/config/json"


def encounter_id_from(arg):
    m = re.search(r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})", arg)
    return m.group(1) if m else arg


def fetch_config():
    with urllib.request.urlopen(CONFIG_URL) as resp:
        data = json.load(resp)
    with open("config.json", "w") as f:
        json.dump(data, f)
    print("Saved config.json")


def fetch_encounter_and_monsters(encounter_id):
    url = f"https://www.dndbeyond.com/encounters/{encounter_id}"
    api_url = f"https://encounter-service.dndbeyond.com/v1/encounters/{encounter_id}"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(storage_state=AUTH_FILE)
        page = context.new_page()

        captured = {}

        def on_response(resp):
            if resp.url == api_url:
                try:
                    captured["encounter"] = resp.json()
                except Exception:
                    pass
            elif "monster-service" in resp.url and "/Monster?" in resp.url:
                try:
                    captured["monsters"] = resp.json()
                except Exception:
                    pass

        page.on("response", on_response)
        page.goto(url, wait_until="networkidle")
        page.wait_for_timeout(1500)

        if "encounter" not in captured:
            raise RuntimeError("Failed to load encounter data — check auth.json is valid and the URL/id is correct.")

        enc_data = captured["encounter"]["data"]
        monster_instances = enc_data["monsters"]

        counts = {}
        for m in monster_instances:
            counts[m["id"]] = counts.get(m["id"], 0) + 1

        ids = list(counts.keys())

        if "monsters" not in captured:
            # The monster-service call only fires when the encounter is "run" or a
            # monster card is expanded client-side; trigger it directly instead.
            ids_qs = "&".join(f"ids={i}" for i in ids)
            monster_api_url = f"https://monster-service.dndbeyond.com/v1/Monster?{ids_qs}"
            resp = page.request.get(monster_api_url)
            captured["monsters"] = resp.json()

        browser.close()

    monsters = captured["monsters"]["data"]
    for m in monsters:
        m["count"] = counts.get(m["id"], 1)

    # keep stable order matching first appearance in the encounter list
    order = {mid: i for i, mid in enumerate(ids)}
    monsters.sort(key=lambda m: order.get(m["id"], 999))

    return {"encounter_name": enc_data["name"], "monsters": monsters}


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run python fetch_stat_blocks.py <encounter_url_or_id>")
        sys.exit(1)

    encounter_id = encounter_id_from(sys.argv[1])
    fetch_config()
    result = fetch_encounter_and_monsters(encounter_id)

    with open("monsters_full.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"Saved monsters_full.json: {result['encounter_name']!r}, {len(result['monsters'])} unique monsters")
    for m in result["monsters"]:
        print(f"  {m['name']} x{m['count']}")


if __name__ == "__main__":
    main()
