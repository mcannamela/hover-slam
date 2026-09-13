"""One-time interactive login. Opens a real browser window, lets you log into
D&D Beyond by hand (including any 2FA), then saves the session to auth.json
so future scripts can reuse it without logging in again.

Uses the real installed Google Chrome (not Playwright's bundled Chromium) with
a dedicated fresh profile, since Google's Sign-In bot detection tends to flag
the bundled Chromium's automation fingerprint.
"""
import os
from playwright.sync_api import sync_playwright

AUTH_FILE = "auth.json"
PROFILE_DIR = os.path.expanduser("~/dndbeyond-organizer/chrome-profile")

def main():
    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            PROFILE_DIR,
            channel="chrome",
            headless=False,
            args=["--disable-blink-features=AutomationControlled"],
        )
        page = context.pages[0] if context.pages else context.new_page()
        page.goto("https://www.dndbeyond.com/login")
        print("Log in in the opened browser window.")
        print("Waiting for a real session cookie to appear (checking every 2s)...")

        def has_session_cookie():
            cookies = context.cookies()
            names = [c["name"] for c in cookies]
            print("  current cookies:", names, "url:", page.url)
            return any(n in names for n in ("CobaltSession", "sessionid", "connect.sid"))

        import time
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            if has_session_cookie():
                break
            page.wait_for_timeout(2000)
        else:
            print("Timed out waiting for session cookie; saving whatever state exists.")

        page.wait_for_timeout(2000)
        context.storage_state(path=AUTH_FILE)
        print(f"Saved session to {AUTH_FILE}")
        context.close()

if __name__ == "__main__":
    main()
