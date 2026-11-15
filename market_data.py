"""
Exhibit D - Market Data Snapshot au 27 novembre 2026.

Toutes les valeurs sont LOCKEES par le cas : on ne tire aucune donnee
d'un feed live. Le seul parametre libre est le spread de credit,
choisi dans la grille ALM (cf. credit_grid).
"""

import numpy as np

# ---------------------------------------------------------------------
# Termes du produit (Exhibit A)
# ---------------------------------------------------------------------
S0 = 2318.47                 # niveau initial de l'indice, fixing 27 nov 2026
NOMINAL = 1000.0             # denomination par note, en EUR
COUPON_RATE = 0.015          # 1.50 % du nominal par trimestre (6 % p.a.)
COUPON_BARRIER = 0.60        # coupon paye si S >= 60 % de S0
AUTOCALL_BARRIER = 1.00      # rappel si S >= 100 % de S0
CAPITAL_BARRIER = 0.60       # capital protege a l'echeance si S >= 60 % de S0
ISSUE_PRICE = 1.00           # emission a 100 % du nominal
MARGIN_FLOOR = 0.015         # plancher de marge brute exige par le desk

# ---------------------------------------------------------------------
# Parametres de marche (Exhibit D) - lockes
# ---------------------------------------------------------------------
VOL_BASE = 0.17              # vol mid du desk, Black plat
DECREMENT = 0.05             # q = 5 % p.a. continu, deja dans l'indice publie

# ---------------------------------------------------------------------
# Calendrier d'observation + facteurs d'actualisation OIS (Table 3)
# (date, T en annees ACT/365.25, DF OIS, autocall possible ?)
# ---------------------------------------------------------------------
SCHEDULE = [
    ("2027-03-15", 0.2957, 0.99305, False),
    ("2027-06-15", 0.5476, 0.98619, False),
    ("2027-09-15", 0.7995, 0.97857, False),
    ("2027-12-15", 1.0486, 0.97035, True),
    ("2028-03-15", 1.2977, 0.96319, False),
    ("2028-06-15", 1.5496, 0.95591, False),
    ("2028-09-15", 1.8015, 0.94841, False),
    ("2028-12-15", 2.0507, 0.94111, True),
    ("2029-03-15", 2.2971, 0.93405, False),
    ("2029-06-15", 2.5489, 0.92709, False),
    ("2029-09-15", 2.8008, 0.91992, False),
    ("2029-12-15", 3.0500, 0.91284, True),
    ("2030-03-15", 3.2964, 0.90614, False),
    ("2030-06-15", 3.5483, 0.89902, False),
    ("2030-09-15", 3.8001, 0.89225, False),
    ("2030-12-15", 4.0493, 0.88525, True),
    ("2031-03-15", 4.2957, 0.87871, False),
    ("2031-06-15", 4.5476, 0.87168, False),
    ("2031-09-15", 4.7995, 0.86507, False),
    ("2031-12-15", 5.0486, 0.85815, False),   # valorisation finale
]

DATES = [row[0] for row in SCHEDULE]
TIMES = np.array([row[1] for row in SCHEDULE])
DF_OIS = np.array([row[2] for row in SCHEDULE])
IS_AUTOCALL = np.array([row[3] for row in SCHEDULE])

N_OBS = len(SCHEDULE)

# ---------------------------------------------------------------------
# Grille de credit ALM (Exhibit D) : spread vs OIS par duree de vie
# ---------------------------------------------------------------------
CREDIT_GRID = {
    1: 0.0010,   # +10 bp - premier autocall juge probable
    2: 0.0015,   # +15 bp - sortie vers l'annee 2
    3: 0.0020,   # +20 bp - la note survit aux premiers rappels
    5: 0.0030,   # +30 bp - financee comme si tenue jusqu'a maturite
}


def forward_rates():
    """
    Taux forwards continus implicites dans la table de DF OIS.

    Entre deux dates d'observation consecutives :
        f = ln(DF_precedent / DF_courant) / (T_courant - T_precedent)

    Ce sont ces forwards qui pilotent le drift de l'indice : la
    couverture se finance a OIS, PAS au taux de la note.
    """
    df_prev = np.concatenate([[1.0], DF_OIS[:-1]])
    t_prev = np.concatenate([[0.0], TIMES[:-1]])
    dt = TIMES - t_prev
    return np.log(df_prev / DF_OIS) / dt, dt


def df_note(spread):
    """
    Facteurs d'actualisation des flux de la NOTE.

        DF_note(T) = DF_OIS(T) * exp(-s * T)

    Le spread ne s'applique qu'ici - jamais au drift de l'indice.
    """
    return DF_OIS * np.exp(-spread * TIMES)
