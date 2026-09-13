import json
import sys
from jinja2 import Template
from playwright.sync_api import sync_playwright
from normalize import load_config, normalize

def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "stat_blocks.pdf"

    with open("monsters_full.json") as f:
        full = json.load(f)
    idx = load_config()

    monsters = [normalize(m, idx) for m in full["monsters"]]

    with open("template.html") as f:
        tmpl = Template(f.read())

    html = tmpl.render(encounter_name=full["encounter_name"], monsters=monsters)
    with open("rendered.html", "w") as f:
        f.write(html)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto("file://" + __file__.rsplit("/", 1)[0] + "/rendered.html")
        page.wait_for_timeout(300)
        page.pdf(path=out_path, print_background=True, format="Letter", landscape=True,
                 margin={"top": "0.2in", "bottom": "0.2in", "left": "0.2in", "right": "0.2in"})
        browser.close()

    print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
