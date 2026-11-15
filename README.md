# Phoenix Autocallable — Monte Carlo Pricer

Monte Carlo pricer for **Phoenix autocallable notes**, with a Streamlit
interface. It computes fair value, derives the issuer's **gross structuring
margin**, and measures how that margin responds to volatility and to each
product lever.

Two modes:

| Mode | What it does |
|---|---|
| **Worked example** | One deal on a frozen market snapshot, reproducible to the cent |
| **Note builder** | Pick an index, set the terms, price any Phoenix |

---

## The product

A Phoenix pays a conditional coupon and can redeem early. Three tests, each
on the index level relative to where it started:

| Test | Frequency | Condition | Outcome |
|---|---|---|---|
| Coupon | quarterly | index ≥ 60 % of initial | pay the coupon, otherwise nothing |
| Autocall | annually | index ≥ 100 % of initial | redeem at par, the note ends |
| Capital | final valuation only | index ≥ 60 % of initial | redeem at par, otherwise `par × index / initial` |

Coupons carry **no memory**: a missed coupon is not recovered later.

A **Phoenix** keeps the coupon and autocall barriers apart, which is what lets
it keep paying in a market that has fallen but not collapsed. An **Athena**
aligns them and accrues coupons to the call date instead.

The capital barrier is a cliff, not a slope. One tick above it the holder is
repaid in full; one tick below, redemption drops to 60 % of nominal and falls
from there with the index.

### Decrement indices

A decrement index deducts a fixed performance — here 5 % a year — by
construction. The mechanism replaces an uncertain future dividend stream with
a known deduction, which lets the issuer hedge without carrying dividend risk.

Under the risk-neutral measure:

```
dS = (r − q)·S·dt + σ·S·dW     with q = the decrement
```

With `r ≈ 3 %` and `q = 5 %` the **drift is negative**. The underlying equities
must outpace the deduction before the index goes anywhere, which makes the
autocall materially harder to reach than the headline terms suggest. Omitting
`q` overstates fair value. The decrement replaces the dividend yield rather
than adding to it.

---

## Method

The GBM is simulated exactly in log space, so there is no discretisation error
from the time step:

```
S(t+Δt) = S(t) · exp[ (f − q − σ²/2)·Δt + σ·√Δt·Z ]
```

`f` is the forward rate implied by the discount curve over each step.

- 50 000 paths by default, **antithetic sampling** to halve the standard error
- Autocall detected at the **first** date breached, with every later cash flow
  switched off
- Note cash flows discounted at the risk-free rate plus an issuer credit
  spread: `DF_note(T) = DF_rf(T) · e^(−sT)`

The spread applies to the note's own flows only. The hedge is funded at the
risk-free rate, so the index drift is unaffected.

### Choosing the credit spread

The spread comes from a funding grid indexed on the note's expected life
(+10 bp at one year, +30 bp at five). A wider spread lowers fair value and
flatters the margin, so the choice needs to stand on its own.

The pricer measures the **simulated expected life** — the mean exit date
across paths — which lets the grid line be read off the model rather than
picked by eye.

---

## Worked example

Terms: 6 % p.a. · 60 % coupon barrier · autocall at 100 % · issued at 100 % ·
17 % implied volatility · 1.50 % margin floor · 5 % decrement index.

Simulated expected life of **2.94 years**, which selects the 3Y grid line at
+20 bp.

| Expected life | Spread | Fair value | Gross margin |
|---|---|---|---|
| 1Y | +10 bp | 993.50 | 0.65 % |
| 2Y | +15 bp | 992.23 | 0.78 % |
| **3Y** | **+20 bp** | **990.97** | **0.90 %** |
| 5Y | +30 bp | 988.44 | 1.16 % |

**No grid line clears the 1.50 % floor.** The proposed terms are uneconomic
for the issuer — and the honest spread choice is not the flattering one.

### What the holder experiences

| | |
|---|---|
| Probability of early redemption | 60.3 % |
| Probability of capital loss | 15.0 % |
| Coupons paid on average | 10.2 of 20 |

The gap between the 6 % headline and what is actually collected is the
autocall: the note is usually redeemed before half its coupons have been paid,
and it redeems in exactly the markets where staying invested would have been
pleasant.

Most of the fair value — around 85 % — is the repayment of capital rather than
the income, which is the reverse of how these notes are usually discussed.

### Volatility repricing risk

| Implied vol | Fair value | Gross margin | vs base |
|---|---|---|---|
| 16 % | 1 000.82 | −0.08 % | −98 bp |
| 17 % | 990.97 | 0.90 % | — |
| 18 % | 981.05 | 1.89 % | +99 bp |

The margin moves roughly **one point of nominal per point of volatility** —
more than the floor itself, and enough to turn the trade loss-making at 16 %.
The economics rest on an assumption that moves daily. That is an argument
about execution timing, not a reason to adopt whichever volatility clears the
floor.

### Revision levers

Each line is the value that brings the margin back to the floor, everything
else unchanged.

| Lever | Breakeven | Cost |
|---|---|---|
| Coupon | 6.00 % → **5.75 %** p.a. | the most visible term to an adviser |
| Coupon barrier | 60 % → **63.6 %** | 9.9 coupons paid instead of 10.2 |
| Capital barrier | 60 % → **62.1 %** | loss probability rises to 16.8 % |

Lowering the coupon or capital barrier makes the note **more** generous and
therefore more expensive to manufacture: those moves cost margin. Raising them
is what recovers it.

---

## Note builder

Pick an index, set the terms, price it. Spot and price history are pulled
live; everything else is yours to set — maturity, coupon frequency, barriers,
autocall schedule, issue price, credit spread, and a synthetic decrement.

### On volatility

The builder offers an **implied volatility field** and a **realised-volatility
proxy**, and labels which one is in use. That distinction is not cosmetic.

Implied volatility is not available from free data sources, and it is the
input that matters most: a one-point error moves the margin by about a point
of nominal. Realised volatility is backward-looking, carries no volatility
risk premium, and typically sits below implied on equity indices — so pricing
on it flatters the margin. It is a reasonable starting point and a poor
substitute.

Decrement indices are not quoted on free sources either. The builder applies a
**synthetic decrement** to a standard index instead, which also makes the
effect easy to see by varying it.

Live quotes are delayed and best-effort. If the provider cannot be reached the
app falls back to manual entry rather than failing.

---

## Running it

```bash
pip install -r requirements.txt

python main.py          # full analysis of the worked example
streamlit run app.py    # interactive interface
```

As a library:

```python
from pricer import price
import worked_example as wx

res = price(wx.SCHEDULE, wx.SPOT, wx.VOL, wx.DECREMENT,
            wx.CREDIT_GRID[3], wx.DEAL)
print(res.fair_value, res.margin_pct, res.expected_life)
```

Any structure, through the schedule builder:

```python
from deal import Deal, build_schedule
from pricer import price

deal = Deal(maturity_years=3, coupons_per_year=2, coupon_rate_pa=0.08,
            coupon_barrier=0.70, autocall_barrier=0.95)
schedule = build_schedule(deal, flat_rate=0.03)
res = price(schedule, spot=5000, vol=0.20, decrement=0.0,
            credit_spread=0.002, deal=deal)
```

### Files

```
deal.py             deal terms, observation schedule construction
pricer.py           path generation and the pricing engine
market.py           live quotes, realised volatility, offline fallback
worked_example.py   the frozen snapshot and its funding grid
charts.py           payoff curve, path fan, distributions, sensitivity
main.py             command-line analysis
app.py              Streamlit interface
```

---

## Limitations

- **Flat Black volatility, no skew.** Barrier products are sensitive to skew;
  a local or stochastic volatility model would give a different number.
- **Lognormal, no jumps** — while tail risk is precisely what the capital
  barrier is exposed to.
- **Flat deterministic credit spread**, applied as a uniform continuous bump.
- No dynamic hedging cost, repo, borrow or market friction.
- Barriers are tested on observation dates only, as the terms specify. Moves
  between dates are not observed, which is a feature of the product rather
  than a modelling shortcut.

Figures are indicative and not a quotation.

## Licence

MIT
