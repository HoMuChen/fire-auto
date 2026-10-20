"""抓 Polymarket 美股/指數每日 Up or Down 市場 + 逐分鐘價格歷史，快取到 data/polymarket/。

events.json : 每個每日市場 {slug, ticker, date, up_token, outcome(1=Up), volume}
prices/{slug}.json : CLOB prices-history（Up token，fidelity=1 分鐘）
"""
import json, os, re, sys, time
from datetime import datetime, timezone
import requests

D = "data/polymarket"
GAMMA = "https://gamma-api.polymarket.com/events"
CLOB = "https://clob.polymarket.com/prices-history"
TICK = re.compile(r"\(([A-Z0-9.^]+)\) Up or Down on")


def list_events():
    seen = {}
    for tag in ("equities", "indicies"):
        cursor, n = None, 0
        while True:
            q = dict(tag_slug=tag, closed="true", limit=100)
            if cursor:
                q["after_cursor"] = cursor
            j = requests.get(GAMMA + "/keyset", params=q, timeout=60).json()
            r = j.get("events", [])
            for e in r:
                m = TICK.search(e.get("title", ""))
                if not m or len(e.get("markets", [])) != 1:
                    continue
                mk = e["markets"][0]
                try:
                    prices = json.loads(mk["outcomePrices"]); toks = json.loads(mk["clobTokenIds"]); outs = json.loads(mk["outcomes"])
                except Exception:
                    continue
                if outs != ["Up", "Down"] or prices not in (["1", "0"], ["0", "1"]):
                    continue  # 未結算 / 50-50
                seen[e["slug"]] = dict(slug=e["slug"], ticker=m.group(1), end=e["endDate"], start=e.get("startDate"),
                                       up_token=toks[0], outcome=1 if prices[0] == "1" else 0,
                                       volume=float(e.get("volume") or 0))
            n += len(r)
            print(tag, n, len(seen), flush=True)
            cursor = j.get("next_cursor")
            if not r or not cursor:
                break
    json.dump(list(seen.values()), open(f"{D}/events.json", "w"))
    return list(seen.values())


def fetch_prices(evs):
    from concurrent.futures import ThreadPoolExecutor
    os.makedirs(f"{D}/prices", exist_ok=True)
    todo = [e for e in evs if not os.path.exists(f"{D}/prices/{e['slug']}.json")]
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda ie: fetch_one(*ie), enumerate(todo)))


def fetch_one(i, e):
    p = f"{D}/prices/{e['slug']}.json"
    end = int(datetime.fromisoformat(e["end"].replace("Z", "+00:00")).timestamp())
    start = end - 3 * 86400
    for _ in range(3):
        try:
            r = requests.get(CLOB, params=dict(market=e["up_token"], startTs=start, endTs=end + 600, fidelity=1), timeout=30)
            h = r.json().get("history", [])
            break
        except Exception:
            time.sleep(2)
    else:
        h = []
    json.dump(h, open(p, "w"))
    if i % 100 == 0:
        print("prices", i, e["slug"], len(h), flush=True)


if __name__ == "__main__":
    evs = list_events() if "skip-list" not in sys.argv else json.load(open(f"{D}/events.json"))
    if "list-only" not in sys.argv:
        fetch_prices(evs)
