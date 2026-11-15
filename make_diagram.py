"""
Build the payoff diagram: decision tree + redemption curve, one page.

    python make_diagram.py

Writes payoff_diagram.svg. The tree shows what is decided on each kind
of observation date; the curve shows redemption at final valuation for
a note that was never called.
"""

from deal import NOMINAL
from pricer import payoff_at_maturity
import worked_example as wx

INK = "#1F2328"
MUTED = "#5A6069"
FAINT = "#8A8F98"
LINE = "#D8DCE1"
BLUE = "#2060C9"
RED = "#BE4A2E"
PANEL = "#FCFCFB"

W, H = 980, 1240

d = wx.DEAL
S0 = wx.SPOT
CPN = d.coupon_amount
CB, AB = d.coupon_barrier, d.autocall_barrier


def box(x, y, w, h, title, body, accent=None, dashed=False):
    stroke = accent or LINE
    dash = ' stroke-dasharray="5 4"' if dashed else ""
    out = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" '
           f'fill="#fff" stroke="{stroke}" stroke-width="1.4"{dash}/>')
    out += (f'<text x="{x+w/2}" y="{y+23}" text-anchor="middle" '
            f'font-size="14.5" font-weight="600" fill="{accent or INK}">'
            f'{title}</text>')
    for i, ln in enumerate(body):
        out += (f'<text x="{x+w/2}" y="{y+44+i*17}" text-anchor="middle" '
                f'font-size="12.5" fill="{MUTED}">{ln}</text>')
    return out


def diamond(cx, cy, w, h, lines):
    pts = f"{cx},{cy-h/2} {cx+w/2},{cy} {cx},{cy+h/2} {cx-w/2},{cy}"
    out = (f'<polygon points="{pts}" fill="{PANEL}" stroke="{FAINT}" '
           f'stroke-width="1.4"/>')
    start = cy - (len(lines) - 1) * 8
    for i, ln in enumerate(lines):
        out += (f'<text x="{cx}" y="{start+i*16+5}" text-anchor="middle" '
                f'font-size="13" font-weight="600" fill="{INK}">{ln}</text>')
    return out


def arrow(x1, y1, x2, y2, label=None, lx=0, ly=0, colour=None):
    c = colour or FAINT
    out = (f'<path d="M {x1} {y1} L {x2} {y2}" stroke="{c}" '
           f'stroke-width="1.4" fill="none" marker-end="url(#a)"/>')
    if label:
        out += (f'<text x="{(x1+x2)/2+lx}" y="{(y1+y2)/2+ly}" '
                f'font-size="12" font-weight="600" fill="{c}" '
                f'text-anchor="middle">{label}</text>')
    return out


def elbow(x1, y1, x2, y2, label=None, colour=None):
    c = colour or FAINT
    mid = (y1 + y2) / 2
    out = (f'<path d="M {x1} {y1} V {mid} H {x2} V {y2}" stroke="{c}" '
           f'stroke-width="1.4" fill="none" marker-end="url(#a)"/>')
    if label:
        out += (f'<text x="{(x1+x2)/2}" y="{mid-7}" font-size="12" '
                f'font-weight="600" fill="{c}" text-anchor="middle">'
                f'{label}</text>')
    return out


# ---------------------------------------------------------------- curve
CX, CY, CW, CH = 118, 790, 760, 330


def px(level):
    return CX + level / 1.60 * CW


def py(value, top):
    return CY + CH - value / top * CH


def curve():
    top = NOMINAL * 1.25
    out = f'<rect x="{CX}" y="{CY}" width="{CW}" height="{CH}" fill="{PANEL}"/>'

    for v in (0, 250, 500, 750, 1000, 1250):
        y = py(v, top)
        out += (f'<line x1="{CX}" y1="{y:.1f}" x2="{CX+CW}" y2="{y:.1f}" '
                f'stroke="{LINE}" stroke-width="1"/>')
        out += (f'<text x="{CX-10}" y="{y+4:.1f}" text-anchor="end" '
                f'font-size="11.5" fill="{FAINT}">{v:,.0f}</text>')

    for lv in (0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6):
        x = px(lv)
        out += (f'<text x="{x:.1f}" y="{CY+CH+20}" text-anchor="middle" '
                f'font-size="11.5" fill="{FAINT}">{lv*100:.0f}%</text>')

    bx = px(CB)
    out += (f'<line x1="{bx:.1f}" y1="{CY}" x2="{bx:.1f}" y2="{CY+CH}" '
            f'stroke="{FAINT}" stroke-width="1.3" stroke-dasharray="5 4"/>')
    out += (f'<text x="{bx+8:.1f}" y="{CY+16}" font-size="12" '
            f'fill="{FAINT}">capital barrier {CB:.0%}</text>')

    lo = [(lv / 100, payoff_at_maturity(d, [lv / 100])[0])
          for lv in range(0, int(CB * 100))]
    hi = [(lv / 100, payoff_at_maturity(d, [lv / 100])[0])
          for lv in range(int(CB * 100), 161)]

    def path(pts):
        return " ".join(f"{'M' if i == 0 else 'L'} {px(l):.1f} "
                        f"{py(v, top):.1f}" for i, (l, v) in enumerate(pts))

    out += (f'<path d="{path(lo)}" stroke="{RED}" stroke-width="2.6" '
            f'fill="none" stroke-linecap="round"/>')
    out += (f'<path d="{path(hi)}" stroke="{BLUE}" stroke-width="2.6" '
            f'fill="none" stroke-linecap="round"/>')

    # the cliff, drawn as the gap it is
    y_lo = py(payoff_at_maturity(d, [CB - 0.001])[0], top)
    y_hi = py(payoff_at_maturity(d, [CB])[0], top)
    out += (f'<circle cx="{bx:.1f}" cy="{y_lo:.1f}" r="4.5" fill="#fff" '
            f'stroke="{RED}" stroke-width="2.2"/>')
    out += (f'<circle cx="{bx:.1f}" cy="{y_hi:.1f}" r="4.5" fill="{BLUE}"/>')
    out += (f'<path d="M {bx+26:.1f} {y_hi+6:.1f} V {y_lo-6:.1f}" '
            f'stroke="{RED}" stroke-width="1.3" stroke-dasharray="3 3" '
            f'marker-end="url(#ar)" marker-start="url(#ar)"/>')
    out += (f'<text x="{bx+34:.1f}" y="{(y_lo+y_hi)/2:.1f}" font-size="12.5" '
            f'font-weight="600" fill="{RED}">401 drop at one index point</text>')

    out += (f'<text x="{px(0.22):.1f}" y="{CY+CH-26}" font-size="12.5" '
            f'font-weight="600" fill="{RED}">capital at risk</text>')
    out += (f'<text x="{px(1.15):.1f}" y="{py(NOMINAL+CPN, top)-16:.1f}" '
            f'font-size="12.5" font-weight="600" fill="{BLUE}">'
            f'capital returned, no upside</text>')

    out += (f'<text x="{CX+CW/2}" y="{CY+CH+46}" text-anchor="middle" '
            f'font-size="12.5" fill="{MUTED}">'
            f'Final index level as a percentage of initial '
            f'({S0:,.2f})</text>')
    out += (f'<text x="{CX-74}" y="{CY+CH/2}" font-size="12.5" '
            f'fill="{MUTED}" transform="rotate(-90 {CX-74} {CY+CH/2})" '
            f'text-anchor="middle">Redemption per 1,000 note</text>')
    return out


# ----------------------------------------------------------------- tree
def tree():
    out = ""
    out += (f'<text x="60" y="152" font-size="15.5" font-weight="600" '
            f'fill="{INK}">What is decided on each observation date</text>')

    out += box(60, 176, 196, 62, "Quarterly date",
               ["20 dates, Mar / Jun", "Sep / Dec 2027-2031"])
    out += box(392, 176, 196, 62, "Annual date",
               ["4 dates, 15 December", "2027 to 2030"])
    out += box(724, 176, 196, 62, "Final valuation",
               ["15 December 2031", "if never called"])

    out += arrow(158, 238, 158, 274)
    out += arrow(490, 238, 490, 274)
    out += arrow(822, 238, 822, 274)

    out += diamond(158, 318, 208, 84, ["Index ≥ 60 %", "of initial?"])
    out += diamond(490, 318, 208, 84, ["Index ≥ 100 %", "of initial?"])
    out += diamond(822, 318, 208, 84, ["Index ≥ 60 %", "of initial?"])

    out += elbow(158, 360, 130, 424, "yes", BLUE)
    out += elbow(158, 360, 278, 424, "no", RED)
    out += elbow(490, 360, 416, 424, "yes", BLUE)
    out += elbow(490, 360, 564, 424, "no", FAINT)
    out += elbow(822, 360, 748, 424, "yes", BLUE)
    out += elbow(822, 360, 896, 424, "no", RED)

    out += box(60, 424, 140, 70, f"Pay {CPN:,.0f}",
               ["1.50 % of nominal"], BLUE)
    out += box(208, 424, 140, 70, "Pay nothing",
               ["lost for good,", "no memory"], RED)
    out += box(346, 424, 140, 70, "Redeem 1,000",
               ["plus coupon due,", "note terminates"], BLUE)
    out += box(494, 424, 140, 70, "Continue",
               ["to the next", "observation date"], FAINT)
    out += box(678, 424, 140, 70, "Redeem 1,000",
               ["plus final coupon"], BLUE)
    out += box(826, 424, 140, 70, "1,000 × S/S₀",
               ["full downside,", "no coupon"], RED)

    out += (f'<path d="M 564 494 V 540 H 34 V 207 H 56" stroke="{FAINT}" '
            f'stroke-width="1.3" stroke-dasharray="5 4" fill="none" '
            f'marker-end="url(#a)"/>')
    out += (f'<text x="330" y="534" text-anchor="middle" font-size="11.5" '
            f'fill="{FAINT}">the cycle repeats until a call or maturity'
            f'</text>')

    rows = [
        ("Underlying",
         "Euro healthcare index, 5 % decrement · initial 2,318.47"),
        ("Coupon barrier", "60 % of initial = 1,391.08 · tested quarterly"),
        ("Autocall barrier", "100 % of initial = 2,318.47 · tested annually"),
        ("Capital barrier", "60 % of initial · tested at final valuation only"),
        ("Coupon", "1.50 % of nominal per quarter, 6.00 % p.a. · no memory"),
    ]
    out += (f'<rect x="60" y="566" width="860" height="{len(rows)*25+22}" '
            f'rx="7" fill="{PANEL}" stroke="{LINE}" stroke-width="1.2"/>')
    for i, (k, v) in enumerate(rows):
        y = 594 + i * 25
        out += (f'<text x="80" y="{y}" font-size="12.5" fill="{FAINT}">'
                f'{k}</text>')
        out += (f'<text x="252" y="{y}" font-size="12.5" fill="{INK}">'
                f'{v}</text>')
    return out


svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}"
     viewBox="0 0 {W} {H}" font-family="Georgia, 'Times New Roman', serif">
  <defs>
    <marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6"
            markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="{FAINT}"/>
    </marker>
    <marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5"
            markerHeight="5" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="{RED}"/>
    </marker>
  </defs>
  <rect width="{W}" height="{H}" fill="#fff"/>

  <text x="60" y="64" font-size="25" font-weight="650" fill="{INK}">
    Healthcare Phoenix Autocallable — payoff map</text>
  <text x="60" y="92" font-size="13.5" fill="{MUTED}">
    HC-PHX-047 · notes due 15 December 2031 · issued at 100 % of
    EUR 1,000 · constructed from Exhibit A</text>
  <line x1="60" y1="112" x2="920" y2="112" stroke="{LINE}"
        stroke-width="1.2"/>

  {tree()}

  <text x="60" y="762" font-size="15.5" font-weight="600" fill="{INK}">
    Redemption at final valuation, if never called</text>

  {curve()}

  <line x1="60" y1="1168" x2="920" y2="1168" stroke="{LINE}"
        stroke-width="1.2"/>
  <text x="60" y="1192" font-size="12" fill="{FAINT}">
    The break at the capital barrier is a discontinuity, not a slope:
    protection applies on the final valuation date only, and only to a
    note that was never called.</text>
  <text x="60" y="1212" font-size="12" fill="{FAINT}">
    Coupons are conditional and carry no memory. Redemption is capped at
    par: the holder has no participation in index gains.</text>
</svg>'''

with open("payoff_diagram.svg", "w") as f:
    f.write(svg)

print("payoff_diagram.svg written")
