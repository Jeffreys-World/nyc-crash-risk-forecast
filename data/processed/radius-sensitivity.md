# Sensitivity of the headline to the three join radii

Snapshot `2026-08-13`. One knob varied at a time; the other two hold at their published values.

Baseline (150 / 100 / 50 ft): R1 48.7%, R2 64.1%, R3 82.5%, lift +18.4pp.

`N` is the size of DOT's list, which every ranking is made to match. `holdout` is the capture-rate denominator - it moves with the corridor join distance because a crash that attaches to no unit at all is outside the universe being scored, so the rates on each row are shares of slightly different totals.

| knob | ft | N | holdout | R1 | R2 | R3 | lift | 95% CI | vs baseline | bar |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: | :---: |
| corridor join distance | 100 | 38,909 | 17,956 | 48.7% | 64.3% | 83.0% | +18.6pp | [+17.7, +19.5] | +0.2pp | yes |
| corridor join distance **(baseline)** | 150 | 38,909 | 18,059 | 48.7% | 64.1% | 82.5% | +18.4pp | [+17.5, +19.3] | +0.0pp | yes |
| corridor join distance | 250 | 38,909 | 18,103 | 48.7% | 64.0% | 82.4% | +18.4pp | [+17.5, +19.2] | -0.1pp | yes |
| intersection radius | 50 | 38,909 | 18,059 | 49.4% | 61.9% | 78.0% | +16.1pp | [+15.2, +16.9] | -2.3pp | yes |
| intersection radius **(baseline)** | 100 | 38,909 | 18,059 | 48.7% | 64.1% | 82.5% | +18.4pp | [+17.5, +19.3] | +0.0pp | yes |
| intersection radius | 150 | 38,909 | 18,059 | 48.3% | 66.5% | 86.4% | +19.9pp | [+19.0, +20.8] | +1.5pp | yes |
| VZV label buffer | 25 | 35,461 | 18,059 | 47.6% | 63.4% | 80.5% | +17.1pp | [+16.3, +18.0] | -1.3pp | yes |
| VZV label buffer **(baseline)** | 50 | 38,909 | 18,059 | 48.7% | 64.1% | 82.5% | +18.4pp | [+17.5, +19.3] | +0.0pp | yes |
| VZV label buffer | 100 | 43,111 | 18,059 | 49.4% | 65.0% | 84.6% | +19.6pp | [+18.8, +20.5] | +1.2pp | yes |

Lift ranges from +16.1pp (intersection radius at 50 ft) to +19.9pp (intersection radius at 150 ft).

Every setting clears the pre-registered 5pp bar.
