# Phoenix Autocallable — Monte Carlo Pricer

Pricer Monte Carlo pour une note **Phoenix autocallable** adossée à un indice
actions **decrement**, avec interface Streamlit.

Le pricer calcule la *fair value* de la note, en déduit la **marge brute de
structuration** de l'émetteur, et mesure la sensibilité de cette marge à la
volatilité et aux paramètres du produit.

---

## Le produit

Une note Phoenix verse un coupon conditionnel et peut être remboursée par
anticipation. Trois mécanismes, testés à des dates d'observation fixées :

| Mécanisme | Fréquence | Condition | Conséquence |
|---|---|---|---|
| Coupon conditionnel | trimestrielle | indice ≥ 60 % du niveau initial | coupon de 1,50 % du nominal |
| Autocall | annuelle | indice ≥ 100 % du niveau initial | remboursement au pair, la note s'éteint |
| Barrière capitale | à l'échéance | indice ≥ 60 % du niveau initial | remboursement au pair, sinon `nominal × S_T / S₀` |

Les coupons sont **sans mémoire** : un coupon manqué n'est pas rattrapé.

Un **Phoenix** sépare la barrière de coupon (60 %) et la barrière d'autocall
(100 %). Dans un **Athena**, les deux coïncident et les coupons sont accumulés
puis payés au rappel.

### L'indice decrement

Le sous-jacent retire une performance fixe de **5 % par an** en continu. Ce
mécanisme remplace l'incertitude sur les dividendes futurs par une déduction
connue d'avance, ce qui permet à l'émetteur de couvrir la note sans porter de
risque de dividende.

Conséquence sur le pricing : sous la mesure risque-neutre,

```
dS = (r − q)·S·dt + σ·S·dW     avec q = 5 %
```

Avec `r ≈ 3 %` et `q = 5 %`, le **drift est négatif**. L'indice dérive vers le
bas, ce qui réduit la probabilité d'autocall et abaisse la fair value.
Omettre `q` surévalue nettement la note. Le decrement remplace le dividende,
il ne s'y ajoute pas.

---

## Méthode

Simulation exacte du GBM entre dates d'observation :

```
S(t+Δt) = S(t) · exp[ (f − q − σ²/2)·Δt + σ·√Δt·Z ]
```

où `f` est le forward OIS implicite dans la table de facteurs d'actualisation.

- 50 000 chemins, **variables antithétiques** pour réduire la variance
- Détection de l'autocall à la **première** date franchie, extinction de tous
  les flux ultérieurs
- Actualisation des flux de la note au taux OIS **majoré d'un spread de
  crédit** : `DF_note(T) = DF_OIS(T) · e^(−sT)`

Le spread ne s'applique qu'aux flux de la note. La couverture se finance à OIS,
donc le drift de l'indice reste piloté par les forwards OIS.

### Choix du spread de crédit

Le spread est lu dans une grille de trésorerie indexée sur la durée de vie
attendue de la note (+10 bp à 1 an, +30 bp à 5 ans). Un spread plus large
abaisse la fair value et flatte donc la marge affichée — le choix doit être
justifié économiquement.

Le pricer mesure la **durée de vie attendue simulée** (moyenne des dates de
sortie), ce qui permet de choisir la ligne de la grille sur la base du modèle
plutôt qu'à l'estime.

---

## Résultats sur le jeu de paramètres d'exemple

Termes : coupon 6 % p.a. · barrière 60 % · autocall 100 % · émission 100 % ·
vol 17 % · plancher de marge 1,50 %.

Durée de vie simulée de **2,94 ans**, qui désigne la ligne 3Y de la grille
(+20 bp).

| Durée retenue | Spread | Fair value | Marge brute |
|---|---|---|---|
| 1Y | +10 bp | 993,50 € | 0,65 % |
| 2Y | +15 bp | 992,23 € | 0,78 % |
| **3Y** | **+20 bp** | **990,97 €** | **0,90 %** |
| 5Y | +30 bp | 988,44 € | 1,16 % |

**Aucun choix de spread n'atteint le plancher de 1,50 %.** Les termes proposés
sont inéconomiques pour l'émetteur.

### Ce que vit le client

| | |
|---|---|
| Probabilité de rappel anticipé | 60,3 % |
| Probabilité de perte en capital | 15,0 % |
| Coupons encaissés en moyenne | 10,2 sur 20 |

L'écart entre le coupon affiché (6 % p.a.) et l'espérance réelle vient de
l'autocall : la note est souvent rappelée avant d'avoir payé la moitié de ses
coupons.

### Risque de repricing en volatilité

| Volatilité | Fair value | Marge | Écart |
|---|---|---|---|
| 16 % | 1 000,82 € | −0,08 % | −98 bp |
| 17 % | 990,97 € | 0,90 % | — |
| 18 % | 981,05 € | 1,89 % | +99 bp |

La marge bouge d'environ **1 point pour 1 point de volatilité**, soit davantage
que le plancher lui-même. Si la volatilité se reprice entre la date de
référence et l'émission, l'économie du deal change entièrement. C'est un
argument de calendrier d'exécution — pas une invitation à retenir l'hypothèse
de volatilité qui arrange.

### Leviers de révision

Chaque ligne donne la valeur du levier qui ramène la marge exactement à 1,50 %,
les autres paramètres restant inchangés.

| Levier | Point mort | Effet sur le client |
|---|---|---|
| Coupon | 6,00 % → **5,75 %** p.a. | le plus visible commercialement |
| Barrière coupon | 60 % → **63,6 %** | coupons plus difficiles, 9,9 payés contre 10,2 |
| Barrière capitale | 60 % → **62,1 %** | protection dégradée, perte en capital à 19 % |

Abaisser la barrière coupon ou la barrière capitale rend le produit **plus
généreux**, donc plus coûteux à fabriquer : ces leviers dégradent la marge. Pour
l'améliorer il faut les **relever**.

---

## Utilisation

```bash
pip install -r requirements.txt

python src/main.py        # tableau de résultats complet
streamlit run app.py      # interface interactive
```

En tant que bibliothèque :

```python
from pricer import price

res = price(expected_life=3)
print(res.fair_value, res.margin_pct, res.expected_life)

# tester un levier
revised = price(expected_life=3, coupon_rate=0.014375)
```

### Structure

```
src/market_data.py   paramètres, calendrier d'observation, courbe, grille de crédit
src/pricer.py        génération des chemins et moteur de pricing
src/main.py          analyse complète en ligne de commande
app.py               interface Streamlit
```

---

## Limites

- Volatilité **Black plate** : pas de surface, donc le skew n'est pas capturé.
  Un produit à barrière est sensible au skew — la fair value serait différente
  avec une volatilité locale ou stochastique.
- Modèle **log-normal** : pas de sauts, alors que le risque de queue est
  précisément ce qui détermine la barrière capitale.
- Spread de crédit **plat et déterministe**, appliqué comme une majoration
  continue uniforme.
- Pas de coût de couverture dynamique, de repo, ni de frottements de marché.

Les paramètres fournis sont un jeu d'exemple cohérent destiné à la
démonstration, pas une cotation de marché.

## Licence

MIT
