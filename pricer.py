"""
Monte Carlo pour le Healthcare Phoenix Autocallable.

Logique, par chemin simule :
  1. generer le niveau de l'indice a chaque date d'observation (GBM, q = 5 %)
  2. parcourir les dates dans l'ordre :
       - coupon si S >= barriere coupon
       - si date d'autocall et S >= barriere autocall : remboursement, STOP
  3. si jamais rappele : barriere finale sur le capital
  4. actualiser chaque flux avec DF_note et sommer

La fair value est la moyenne de ces valeurs actualisees.
"""

from dataclasses import dataclass, field

import numpy as np

from market_data import (
    AUTOCALL_BARRIER,
    CAPITAL_BARRIER,
    COUPON_BARRIER,
    COUPON_RATE,
    CREDIT_GRID,
    DECREMENT,
    IS_AUTOCALL,
    ISSUE_PRICE,
    MARGIN_FLOOR,
    N_OBS,
    NOMINAL,
    S0,
    TIMES,
    VOL_BASE,
    df_note,
    forward_rates,
)


@dataclass
class PricingResult:
    """Sortie d'un pricing : valeur, marge, et diagnostics."""

    fair_value: float            # EUR par note
    margin_pct: float            # marge brute en % du nominal
    passes_floor: bool
    expected_life: float         # duree de vie moyenne, en annees
    prob_autocall: float         # proba de rappel avant maturite
    prob_capital_loss: float     # proba de finir sous la barriere capitale
    avg_coupons_paid: float      # nombre moyen de coupons encaisses
    pv_coupons: float            # part de la FV venant des coupons
    pv_redemption: float         # part de la FV venant du remboursement
    inputs: dict = field(default_factory=dict)


def simulate_paths(n_paths, vol, seed=42):
    """
    Genere les niveaux de l'indice aux N_OBS dates d'observation.

    Schema exact du GBM entre deux dates (pas de discretisation d'Euler,
    le GBM se simule exactement) :

        S(t+dt) = S(t) * exp[ (f - q - vol^2/2)*dt + vol*sqrt(dt)*Z ]

    ou f est le forward OIS sur la periode et q le decrement de 5 %.

    Retourne un tableau (n_paths, N_OBS).
    """
    fwd, dt = forward_rates()

    rng = np.random.default_rng(seed)
    # Variables antithetiques : pour chaque tirage Z on simule aussi -Z.
    # Cela reduit la variance sans cout supplementaire.
    half = n_paths // 2
    z_half = rng.standard_normal((half, N_OBS))
    z = np.concatenate([z_half, -z_half], axis=0)

    drift = (fwd - DECREMENT - 0.5 * vol**2) * dt
    diffusion = vol * np.sqrt(dt)

    log_increments = drift + diffusion * z
    log_path = np.cumsum(log_increments, axis=1)

    return S0 * np.exp(log_path)


def price(n_paths=50_000, vol=VOL_BASE, expected_life=5,
          coupon_rate=COUPON_RATE, coupon_barrier=COUPON_BARRIER,
          autocall_barrier=AUTOCALL_BARRIER, capital_barrier=CAPITAL_BARRIER,
          issue_price=ISSUE_PRICE, seed=42):
    """
    Price le Phoenix et renvoie fair value + marge + diagnostics.

    Les parametres produit sont exposes pour permettre les revisions
    de l'Option B (changer UN levier et repricer).
    """
    spread = CREDIT_GRID[expected_life]
    disc = df_note(spread)

    paths = simulate_paths(n_paths, vol, seed)
    n = paths.shape[0]

    coupon_amt = coupon_rate * NOMINAL
    lvl_coupon = coupon_barrier * S0
    lvl_autocall = autocall_barrier * S0
    lvl_capital = capital_barrier * S0

    # --- determiner la date de sortie de chaque chemin -----------------
    # Un chemin est rappele a la PREMIERE date d'autocall ou S >= 100 %.
    autocall_idx = np.where(IS_AUTOCALL)[0]
    triggered = paths[:, autocall_idx] >= lvl_autocall       # (n, 4)

    any_call = triggered.any(axis=1)
    # argmax renvoie l'indice du premier True ; sans True il renvoie 0,
    # d'ou le masque any_call applique ensuite.
    first_call_pos = np.argmax(triggered, axis=1)
    exit_idx = np.where(any_call, autocall_idx[first_call_pos], N_OBS - 1)

    # --- coupons -------------------------------------------------------
    # Un coupon est paye si (a) la date est <= date de sortie et
    #                       (b) l'indice y est >= barriere coupon.
    obs_grid = np.arange(N_OBS)[None, :]
    alive = obs_grid <= exit_idx[:, None]
    coupon_paid = alive & (paths >= lvl_coupon)

    pv_coupons_per_path = (coupon_paid * coupon_amt * disc[None, :]).sum(axis=1)

    # --- remboursement -------------------------------------------------
    final_level = paths[:, -1]
    redemption = np.where(
        any_call,
        NOMINAL,                                      # rappele : au pair
        np.where(final_level >= lvl_capital,
                 NOMINAL,                             # au-dessus barriere
                 NOMINAL * final_level / S0),         # sous barriere
    )
    pv_redemption_per_path = redemption * disc[exit_idx]

    # --- agregation ----------------------------------------------------
    pv_per_path = pv_coupons_per_path + pv_redemption_per_path
    fair_value = pv_per_path.mean()

    margin = issue_price - fair_value / NOMINAL

    return PricingResult(
        fair_value=fair_value,
        margin_pct=margin,
        passes_floor=margin >= MARGIN_FLOOR,
        expected_life=float(TIMES[exit_idx].mean()),
        prob_autocall=float(any_call.mean()),
        prob_capital_loss=float((~any_call & (final_level < lvl_capital)).mean()),
        avg_coupons_paid=float(coupon_paid.sum(axis=1).mean()),
        pv_coupons=float(pv_coupons_per_path.mean()),
        pv_redemption=float(pv_redemption_per_path.mean()),
        inputs={
            "n_paths": n, "vol": vol, "expected_life": expected_life,
            "spread_bp": spread * 1e4, "coupon_rate": coupon_rate,
            "coupon_barrier": coupon_barrier,
            "autocall_barrier": autocall_barrier,
            "capital_barrier": capital_barrier, "issue_price": issue_price,
        },
    )
