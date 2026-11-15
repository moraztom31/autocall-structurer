"""
Charts for the pricer interface.

House rules kept throughout: one hue per series, no second y-axis, a
legend whenever more than one series is drawn, recessive grid lines,
and a tooltip on every mark. Colour carries meaning only where the
meaning is real - capital at risk against capital returned - and is
always backed by a label, never left to carry identity on its own.
"""

import altair as alt
import numpy as np
import pandas as pd

# Two hues, far enough apart to survive the common colour-vision
# deficiencies, plus a neutral for reference lines and context marks.
PROTECTED = "#2060C9"     # capital returned in full
AT_RISK = "#BE4A2E"       # capital reduced with the index
NEUTRAL = "#8A8F98"
GRID = "#E3E6EA"

_AXIS = alt.Axis(grid=True, gridColor=GRID, gridOpacity=0.7,
                 domainColor=GRID, tickColor=GRID, labelColor="#5A6069",
                 titleColor="#3C4149", labelFontSize=11, titleFontSize=12)


def _base(df, height=320):
    return alt.Chart(df).properties(height=height).configure_view(
        strokeWidth=0
    ).configure_legend(
        labelColor="#3C4149", titleColor="#3C4149",
        labelFontSize=11, titleFontSize=11, orient="top-left",
    )


def payoff_curve(deal, payoff_fn, height=340):
    """
    Redemption at final valuation against the final index level.

    Drawn for a note that was never called, which is the only path on
    which the capital barrier is ever tested. The curve is deliberately
    broken at the barrier: the drop there is a cliff, not a slope, and
    joining the two segments would draw a transition that does not exist.
    """
    barrier = deal.capital_barrier

    below = np.linspace(0.0, barrier - 1e-6, 160)
    above = np.linspace(barrier, 1.60, 160)

    df = pd.concat([
        pd.DataFrame({"level": below, "payout": payoff_fn(deal, below),
                      "region": "Capital at risk"}),
        pd.DataFrame({"level": above, "payout": payoff_fn(deal, above),
                      "region": "Capital returned"}),
    ])
    df["level_pct"] = df["level"] * 100

    line = alt.Chart(df).mark_line(strokeWidth=2.5).encode(
        x=alt.X("level_pct:Q", title="Final index level (% of initial)",
                scale=alt.Scale(domain=[0, 160]), axis=_AXIS),
        y=alt.Y("payout:Q", title="Redemption per 1,000 note",
                scale=alt.Scale(domain=[0, 1150]), axis=_AXIS),
        color=alt.Color("region:N", title=None,
                        scale=alt.Scale(domain=["Capital at risk",
                                                "Capital returned"],
                                        range=[AT_RISK, PROTECTED])),
        detail="region:N",
        tooltip=[alt.Tooltip("level_pct:Q", title="Index", format=".0f"),
                 alt.Tooltip("payout:Q", title="Redemption", format=",.0f"),
                 alt.Tooltip("region:N", title="")],
    )

    rule = alt.Chart(pd.DataFrame({"x": [barrier * 100]})).mark_rule(
        color=NEUTRAL, strokeDash=[5, 4], strokeWidth=1.5
    ).encode(x="x:Q")

    label = alt.Chart(pd.DataFrame({
        "x": [barrier * 100], "y": [1100],
        "t": [f"Capital barrier {barrier:.0%}"],
    })).mark_text(align="left", dx=7, fontSize=11, color=NEUTRAL).encode(
        x="x:Q", y="y:Q", text="t:N")

    return (rule + line + label).properties(height=height).configure_view(
        strokeWidth=0
    ).configure_legend(
        labelColor="#3C4149", labelFontSize=11, orient="top-left",
        symbolType="stroke", symbolStrokeWidth=3,
    )


def path_fan(paths, schedule, spot, deal, n_shown=140, height=360):
    """
    A sample of simulated index paths, with the two barriers marked.

    Thin, heavily transparent lines: the shape of the cloud is the point,
    not any single trajectory. The downward tilt is the decrement.
    """
    n_shown = min(n_shown, paths.shape[0])
    sample = paths[:n_shown]

    df = pd.DataFrame({
        "t": np.tile(schedule.times, n_shown),
        "level": sample.ravel(),
        "path": np.repeat(np.arange(n_shown), len(schedule)),
    })

    lines = alt.Chart(df).mark_line(strokeWidth=0.6, opacity=0.22,
                                    color=PROTECTED).encode(
        x=alt.X("t:Q", title="Years from trade date", axis=_AXIS),
        y=alt.Y("level:Q", title="Index level", axis=_AXIS),
        detail="path:N",
    )

    marks = pd.DataFrame({
        "y": [deal.autocall_barrier * spot, deal.coupon_barrier * spot],
        "label": [f"Autocall {deal.autocall_barrier:.0%}",
                  f"Coupon / capital {deal.coupon_barrier:.0%}"],
    })
    rules = alt.Chart(marks).mark_rule(
        strokeDash=[5, 4], strokeWidth=1.5, color=NEUTRAL
    ).encode(y="y:Q", tooltip=["label:N"])

    text = alt.Chart(marks).mark_text(
        align="left", dx=6, dy=-7, fontSize=11, color=NEUTRAL
    ).encode(y="y:Q", text="label:N",
             x=alt.value(8))

    return (lines + rules + text).properties(height=height).configure_view(
        strokeWidth=0)


def payout_distribution(total_payout, issue_amount=1000.0, height=320):
    """
    Undiscounted cash returned per note across simulated paths.

    The shape matters more than any summary statistic: a Phoenix pays a
    narrow band of good outcomes very often and a long left tail rarely,
    which is exactly what an average return conceals.
    """
    df = pd.DataFrame({"payout": total_payout})
    df["region"] = np.where(df["payout"] < issue_amount,
                            "Below issue price", "At or above issue price")

    chart = alt.Chart(df).mark_bar(opacity=0.9).encode(
        x=alt.X("payout:Q", bin=alt.Bin(maxbins=70),
                title="Total cash returned per 1,000 note", axis=_AXIS),
        y=alt.Y("count():Q", title="Simulated paths", axis=_AXIS),
        color=alt.Color("region:N", title=None,
                        scale=alt.Scale(
                            domain=["Below issue price",
                                    "At or above issue price"],
                            range=[AT_RISK, PROTECTED])),
        tooltip=[alt.Tooltip("count():Q", title="Paths"),
                 alt.Tooltip("region:N", title="")],
    )
    return chart.properties(height=height).configure_view(
        strokeWidth=0
    ).configure_legend(labelColor="#3C4149", labelFontSize=11,
                       orient="top-right")


def exit_distribution(exit_times, schedule, height=300):
    """
    When the note leaves: each autocall date, then scheduled maturity.

    The last bar lumps together every path that ran the full term, called
    or not, so it is always the tallest when calls are unlikely.
    """
    uniq, counts = np.unique(np.round(exit_times, 4), return_counts=True)
    share = counts / counts.sum()

    labels = []
    for t in uniq:
        idx = int(np.argmin(np.abs(schedule.times - t)))
        tag = "maturity" if idx == len(schedule) - 1 else "autocall"
        labels.append(f"{schedule.labels[idx]}  ({tag})")

    df = pd.DataFrame({"when": labels, "share": share, "paths": counts})

    bars = alt.Chart(df).mark_bar(
        color=PROTECTED, cornerRadiusEnd=4, size=26
    ).encode(
        y=alt.Y("when:N", title=None, sort=None, axis=_AXIS),
        x=alt.X("share:Q", title="Share of simulated paths",
                axis=alt.Axis(format="%", grid=True, gridColor=GRID,
                              gridOpacity=0.7, domainColor=GRID,
                              tickColor=GRID, labelColor="#5A6069",
                              titleColor="#3C4149", labelFontSize=11,
                              titleFontSize=12)),
        tooltip=[alt.Tooltip("when:N", title="Exit"),
                 alt.Tooltip("share:Q", title="Share", format=".1%"),
                 alt.Tooltip("paths:Q", title="Paths", format=",")],
    )

    labels_layer = alt.Chart(df).mark_text(
        align="left", dx=6, fontSize=11, color="#5A6069"
    ).encode(y=alt.Y("when:N", sort=None), x="share:Q",
             text=alt.Text("share:Q", format=".0%"))

    return (bars + labels_layer).properties(height=height).configure_view(
        strokeWidth=0)


def vol_sensitivity(rows, floor, height=300):
    """
    Structuring margin against implied volatility, with the floor drawn.

    The slope is the message. On this structure the margin moves about a
    point of nominal per point of volatility, so where the curve crosses
    the floor decides whether the trade exists at all.
    """
    df = pd.DataFrame(rows)
    df["vol_pct"] = df["vol"] * 100
    df["margin_pct"] = df["margin"] * 100

    line = alt.Chart(df).mark_line(
        strokeWidth=2.5, color=PROTECTED, point=alt.OverlayMarkDef(
            size=70, color=PROTECTED)
    ).encode(
        x=alt.X("vol_pct:Q", title="Implied volatility (%)",
                scale=alt.Scale(zero=False), axis=_AXIS),
        y=alt.Y("margin_pct:Q", title="Gross structuring margin (%)",
                axis=_AXIS),
        tooltip=[alt.Tooltip("vol_pct:Q", title="Vol", format=".1f"),
                 alt.Tooltip("margin_pct:Q", title="Margin", format=".2f")],
    )

    floor_rule = alt.Chart(pd.DataFrame({"y": [floor * 100]})).mark_rule(
        color=AT_RISK, strokeDash=[5, 4], strokeWidth=1.5).encode(y="y:Q")

    floor_text = alt.Chart(pd.DataFrame({
        "y": [floor * 100], "t": [f"Margin floor {floor:.2%}"],
    })).mark_text(align="left", dy=-8, fontSize=11, color=AT_RISK).encode(
        y="y:Q", text="t:N", x=alt.value(8))

    return (floor_rule + line + floor_text).properties(
        height=height).configure_view(strokeWidth=0)
