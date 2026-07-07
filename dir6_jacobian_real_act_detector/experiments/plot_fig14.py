"""fig14 — Phase 7 steering-repair Pareto frontiers (PIL; env has no matplotlib).

Two panels sharing the achieved-steering-effect E on the x-axis:
  (a) E vs off-target collateral C   (b) E vs KL-from-clean.
Curves: alpha_shrink (the 'merely shrink alpha' control), manifold(t) (Phase-6b repair), random(t).
Lower y at a given E = better validity for the same intended effect. alpha_shrink lying BELOW manifold
is the headline null: for a structured steering edit, shrinking alpha preserves the effect/validity
tradeoff better than the manifold projection.
"""
import csv, os
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results"); OUT = os.path.join(HERE, "..", "plots")
F = lambda s: ImageFont.load_default(s)
rows = list(csv.DictReader(open(os.path.join(RES, "steering_repair_metrics.csv"))))
D = {r["method"]: r for r in rows}


def series(prefix, key):
    pts = []
    for r in rows:
        if r["method"].startswith(prefix):
            pts.append((float(r["E_effect"]), float(r[key])))
    return sorted(pts)


def panel(d, ox, oy, pw, ph, ykey, ylabel, title):
    shrink = series("alpha_shrink", ykey)
    manif = series("manifold", ykey)
    rand = series("random", ykey)
    full = (float(D["steered(full)"]["E_effect"]), float(D["steered(full)"][ykey]))
    allpts = shrink + manif + rand + [full]
    xs = [p[0] for p in allpts]; ys = [p[1] for p in allpts]
    x0, x1 = 0, max(xs) * 1.05; y0, y1 = 0, max(ys) * 1.08
    def X(v): return ox + pw * (v - x0) / (x1 - x0)
    def Y(v): return oy + ph * (1 - (v - y0) / (y1 - y0))
    d.text((ox + pw / 2, oy - 24), title, font=F(19), fill=(0, 0, 0), anchor="ma")
    for i in range(6):
        yy = oy + ph * i / 5; vv = y1 - (y1 - y0) * i / 5
        d.line([(ox, yy), (ox + pw, yy)], fill=(233, 233, 233))
        d.text((ox - 8, yy), f"{vv:.0f}" if y1 > 5 else f"{vv:.2f}", font=F(13), fill=(90, 90, 90), anchor="rm")
    for i in range(6):
        xx = ox + pw * i / 5; vv = x0 + (x1 - x0) * i / 5
        d.line([(xx, oy), (xx, oy + ph)], fill=(243, 243, 243))
        d.text((xx, oy + ph + 8), f"{vv:.0f}", font=F(13), fill=(90, 90, 90), anchor="ma")
    d.line([(ox, oy), (ox, oy + ph)], fill=(0, 0, 0)); d.line([(ox, oy + ph), (ox + pw, oy + ph)], fill=(0, 0, 0))
    d.text((ox + pw / 2, oy + ph + 30), "achieved steering effect  E = <dL, d_hat>", font=F(14), fill=(40, 40, 40), anchor="ma")
    d.text((ox - 52, oy + ph / 2), ylabel, font=F(14), fill=(40, 40, 40), anchor="mm")

    def draw(pts, col, r=5, line=True):
        if line:
            for (a, b) in zip(pts, pts[1:]):
                d.line([(X(a[0]), Y(a[1])), (X(b[0]), Y(b[1]))], fill=col, width=3)
        for (x, y) in pts:
            d.ellipse([X(x) - r, Y(y) - r, X(x) + r, Y(y) + r], fill=col, outline=(255, 255, 255))
    draw(rand, (150, 150, 150), r=4, line=False)
    draw(shrink, (31, 119, 180))
    draw(manif, (214, 39, 40))
    # full-steer marker
    d.ellipse([X(full[0]) - 6, Y(full[1]) - 6, X(full[0]) + 6, Y(full[1]) + 6], outline=(0, 0, 0), width=2)


def legend(d, x, y):
    items = [((31, 119, 180), "alpha_shrink (shrink coefficient — control)"),
             ((214, 39, 40), "manifold(t) (Phase-6b kNN repair)"),
             ((150, 150, 150), "random(t)-matched (direction control)"),
             ((0, 0, 0), "full steer (start)")]
    for i, (c, lab) in enumerate(items):
        yy = y + i * 22
        d.ellipse([x, yy, x + 12, yy + 12], fill=c if i < 3 else (255, 255, 255), outline=c)
        d.text((x + 20, yy + 6), lab, font=F(14), fill=(0, 0, 0), anchor="lm")


W, H = 1280, 660
img = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(img)
d.text((W / 2, 16), "Phase 7 — steering repair: manifold projection vs shrinking alpha (matched achieved effect)",
       font=F(20), fill=(0, 0, 0), anchor="ma")
panel(d, 90, 88, 480, 372, "C_offtarget", "off-target collateral C", "(a) off-target output collateral")
panel(d, 720, 88, 480, 372, "KL_from_clean", "KL(clean || x)", "(b) KL from clean output")
legend(d, 700, 512)
img.save(os.path.join(OUT, "fig14_steering_repair.png"))
print("wrote fig14_steering_repair.png")
