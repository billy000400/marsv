"""Manual reading of the 100 widest tokens (W_final > 0.3 side) -> results/france_widest100_manual.csv.

Labels are the author's reading of each decoded token, assigned by hand (no automated labeling):
  place = place name or an unambiguous piece of one (e.g. 'agascar', 'raltar', ' Rica')
  scraped = code-like, username-like or byte/foreign-script token of the kind dir27 called rarely seen
  other = anything else, including names and fragments whose reading is ambiguous (' Lydia', 'eden')
"""
import csv
import os

from analyze import RES

SCRAPED = {2, 5, 9, 14, 16, 28, 32, 35, 40, 47, 48, 51, 52, 53, 73, 81, 82}
PLACE = {1, 3, 4, 6, 7, 8, 10, 12, 13, 15, 17, 18, 20, 21, 24, 26, 29, 30, 31, 33, 36, 37, 41, 43, 44, 45,
         46, 50, 55, 59, 60, 62, 63, 65, 66, 67, 68, 69, 70, 74, 76, 78, 84, 86, 87, 88, 89, 90, 92, 93,
         95, 96, 98}


def main():
    rows = list(csv.DictReader(open(os.path.join(RES, "france_W_gt_0.3.csv"))))[:100]
    with open(os.path.join(RES, "france_widest100_manual.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "token_id", "token", "W_final", "manual_label"])
        for r in rows:
            k = int(r["rank"])
            lab = "place" if k in PLACE else "scraped" if k in SCRAPED else "other"
            w.writerow([k, r["token_id"], r["token"], r["W_final"], lab])
    print({lab: sum((int(r["rank"]) in s) for r in rows) for lab, s in (("place", PLACE), ("scraped", SCRAPED))})


if __name__ == "__main__":
    main()
