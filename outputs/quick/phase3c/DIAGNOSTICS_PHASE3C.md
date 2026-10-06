# Phase 3c — artefact check of D4 / D5 (**POST-HOC**, diagnostic only)

_Run mode **quick**. **POST-HOC**: designed after the Phase 3b results were seen; nothing here was pre-registered. No model, attacker, threshold or split was changed and nothing was retrained. The TEST split is re-used post hoc with the frozen Phase 3b weights and the frozen τ10 (letters-only: τ10 recomputed on val humans for the new masked score). Freeze guards (Phases 1, 2, 3a, 3b pre-registration and `docs/phase3b_frozen_hashes.json`) pass before and after the run. Re-scoring S1/S2 reproduces the stored Phase 3b scores: max |diff| = 0. Device cuda (NVIDIA GeForce RTX 3050 6GB Laptop GPU)._

Question: are D4's and D5's detections of A2–A5b driven by timing given the keys (a real signal), or by differences in the key sequences the bots generate (an implementation artefact)?

## 1. Keycode realism (test humans vs test bots, raw sequences before truncation)

| group | sessions | length mean / median [p5, p95] | UNK % | Backspace / 100 keys | Delete / 100 keys | sessions with Bksp/Del | CapsLock / 100 keys | non-letter/space share |
|---|---|---|---|---|---|---|---|---|
| humans | 4,344 | 48.7 / 47 [22, 80] | 0.023 | 5.43 | 0.003 | 64.8% | 0.59 | 12.5% |
| A1 | 4,344 | 43.3 / 42 [21, 68] | 0.000 | 0.00 | 0.000 | 0.0% | 0.00 | 7.2% |
| A2 | 4,344 | 43.3 / 42 [21, 68] | 0.000 | 0.00 | 0.000 | 0.0% | 0.00 | 7.2% |
| A3 | 4,344 | 43.3 / 42 [21, 68] | 0.000 | 0.00 | 0.000 | 0.0% | 0.00 | 7.2% |
| A4 | 4,344 | 48.0 / 46 [22, 79] | 0.000 | 4.76 | 0.000 | 55.3% | 0.00 | 11.5% |
| A5a | 4,344 | 48.5 / 46 [23, 81] | 0.025 | 5.43 | 0.005 | 62.7% | 0.46 | 12.7% |
| A5b | 4,344 | 43.3 / 42 [21, 68] | 0.000 | 0.00 | 0.000 | 0.0% | 0.00 | 7.2% |

**JSD** (Jensen–Shannon divergence, base 2, 0 = identical, 1 = disjoint) of the keycode and keycode-bigram distributions, test humans vs each attacker:

| attacker | jsd_keycode | jsd_bigram |
|---|---|---|
| A1 | 0.0328 | 0.0557 |
| A2 | 0.0328 | 0.0557 |
| A3 | 0.0328 | 0.0557 |
| A4 | 0.006 | 0.0205 |
| A5a | 0.0006 | 0.0065 |
| A5b | 0.0328 | 0.0555 |

**Shift pattern.** Capitals are counted in the target sentence (both sides type the same sentence); 'Shift-char' also counts punctuation that needs Shift. Position = the next non-Shift key after each Shift press.

| group | Shift per capital | Shift per Shift-char | next key a letter | Shift released before next press | released during next key | released after next key | Shift→key lead (median ms) | Shift hold median [p10, p90] ms |
|---|---|---|---|---|---|---|---|---|
| humans | 0.994 | 0.930 | 90.7% | 5.4% | 64.1% | 30.5% | 159 | 243 [140, 547] |
| A1 | 1.068 | 1.000 | 93.6% | 0.0% | 0.0% | 100.0% | 239 | 385 [240, 533] |
| A2 | 1.068 | 1.000 | 93.6% | 0.0% | 80.4% | 19.6% | 136 | 217 [117, 472] |
| A3 | 1.068 | 1.000 | 93.6% | 0.0% | 80.7% | 19.3% | 137 | 220 [117, 480] |
| A4 | 1.122 | 1.050 | 93.7% | 0.0% | 76.9% | 23.1% | 138 | 224 [120, 519] |
| A5a | 1.106 | 1.035 | 90.8% | 10.2% | 62.8% | 27.1% | 156 | 224 [124, 516] |
| A5b | 1.068 | 1.000 | 93.6% | 36.6% | 49.6% | 13.8% | 162 | 160 [79, 404] |

**Keys other than letters and space** (punctuation, digits, control keys), per 1,000 keystrokes; keycodes present in ≥ 0.1% of the sessions of at least one group:

| keycode | name | humans | A1 | A2 | A3 | A4 | A5a | A5b |
|---|---|---|---|---|---|---|---|---|
| 8 | Backspace | 54.35 | 0.00 | 0.00 | 0.00 | 47.59 | 54.34 | 0.00 |
| 16 | Shift | 31.78 | 38.41 | 38.41 | 38.41 | 36.41 | 35.54 | 38.41 |
| 190 | . | 17.60 | 20.76 | 20.76 | 20.76 | 19.27 | 18.24 | 20.76 |
| 20 | CapsLock | 5.90 | 0.00 | 0.00 | 0.00 | 0.00 | 4.56 | 0.00 |
| 222 | ' | 3.75 | 4.27 | 4.27 | 4.27 | 3.94 | 3.68 | 4.27 |
| 188 | , | 3.18 | 2.73 | 2.73 | 2.73 | 2.52 | 2.87 | 2.73 |
| 191 | / | 2.28 | 2.15 | 2.15 | 2.15 | 1.99 | 2.14 | 2.15 |
| 49 | digit 1 | 0.78 | 0.92 | 0.92 | 0.92 | 0.86 | 0.59 | 0.92 |
| 37 | Left | 0.70 | 0.00 | 0.00 | 0.00 | 0.00 | 0.85 | 0.00 |
| 189 | - | 0.59 | 0.75 | 0.75 | 0.75 | 0.69 | 0.50 | 0.75 |
| 48 | digit 0 | 0.48 | 0.56 | 0.56 | 0.56 | 0.50 | 0.38 | 0.56 |
| 186 | ; | 0.47 | 0.00 | 0.00 | 0.00 | 0.00 | 0.57 | 0.00 |
| 187 | = | 0.47 | 0.00 | 0.00 | 0.00 | 0.00 | 0.22 | 0.00 |
| 50 | digit 2 | 0.27 | 0.29 | 0.29 | 0.29 | 0.26 | 0.23 | 0.29 |
| 39 | Right | 0.27 | 0.00 | 0.00 | 0.00 | 0.00 | 0.25 | 0.00 |
| 220 | \ | 0.24 | 0.00 | 0.00 | 0.00 | 0.00 | 0.08 | 0.00 |
| 17 | Ctrl | 0.22 | 0.00 | 0.00 | 0.00 | 0.00 | 0.05 | 0.00 |
| 51 | digit 3 | 0.20 | 0.22 | 0.22 | 0.22 | 0.20 | 0.23 | 0.22 |
| 57 | digit 9 | 0.18 | 0.21 | 0.21 | 0.21 | 0.19 | 0.21 | 0.21 |
| 52 | digit 4 | 0.14 | 0.16 | 0.16 | 0.16 | 0.14 | 0.19 | 0.16 |
| 53 | digit 5 | 0.14 | 0.16 | 0.16 | 0.16 | 0.15 | 0.19 | 0.16 |
| 192 | ` | 0.14 | 0.00 | 0.00 | 0.00 | 0.00 | 0.21 | 0.00 |
| 54 | digit 6 | 0.11 | 0.14 | 0.14 | 0.14 | 0.13 | 0.09 | 0.14 |
| 55 | digit 7 | 0.09 | 0.13 | 0.13 | 0.13 | 0.12 | 0.10 | 0.13 |
| 219 | [ | 0.09 | 0.00 | 0.00 | 0.00 | 0.00 | 0.04 | 0.00 |
| 173 | - (FF) | 0.09 | 0.00 | 0.00 | 0.00 | 0.00 | 0.08 | 0.00 |
| 97 | keycode 97 | 0.09 | 0.00 | 0.00 | 0.00 | 0.00 | 0.10 | 0.00 |
| 96 | keycode 96 | 0.08 | 0.00 | 0.00 | 0.00 | 0.00 | 0.13 | 0.00 |
| 56 | digit 8 | 0.08 | 0.08 | 0.08 | 0.08 | 0.07 | 0.06 | 0.08 |
| 59 | keycode 59 | 0.06 | 0.00 | 0.00 | 0.00 | 0.00 | 0.05 | 0.00 |
| 99 | keycode 99 | 0.06 | 0.00 | 0.00 | 0.00 | 0.00 | 0.05 | 0.00 |
| 18 | Alt | 0.05 | 0.00 | 0.00 | 0.00 | 0.00 | 0.02 | 0.00 |
| 105 | keycode 105 | 0.04 | 0.00 | 0.00 | 0.00 | 0.00 | 0.02 | 0.00 |
| 45 | Insert | 0.04 | 0.00 | 0.00 | 0.00 | 0.00 | 0.04 | 0.00 |
| 98 | keycode 98 | 0.04 | 0.00 | 0.00 | 0.00 | 0.00 | 0.04 | 0.00 |
| 102 | keycode 102 | 0.04 | 0.00 | 0.00 | 0.00 | 0.00 | 0.04 | 0.00 |
| 61 | keycode 61 | 0.03 | 0.00 | 0.00 | 0.00 | 0.00 | 0.03 | 0.00 |
| 46 | Delete | 0.03 | 0.00 | 0.00 | 0.00 | 0.00 | 0.05 | 0.00 |
| 35 | End | 0.03 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | keycode 100 | 0.03 | 0.00 | 0.00 | 0.00 | 0.00 | 0.03 | 0.00 |
| 110 | keycode 110 | 0.02 | 0.00 | 0.00 | 0.00 | 0.00 | 0.02 | 0.00 |
| 36 | Home | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.06 | 0.00 |
| 33 | PageUp | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.02 | 0.00 |
| 91 | Meta | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.06 | 0.00 |

**Keycodes ≥ 10× more frequent in one group** (frequency = share of all keystrokes; present in ≥ 0.1% of the sessions of the group where it is more frequent; ∞ = absent from the other group):

| attacker | more_in | name | freq_humans | freq_bots | ratio | sessions_humans | sessions_bots |
|---|---|---|---|---|---|---|---|
| A1 | humans | Backspace | 5.435% | 0.000% | ∞ | 64.8% | 0.0% |
| A1 | humans | CapsLock | 0.590% | 0.000% | ∞ | 9.7% | 0.0% |
| A1 | humans | Left | 0.070% | 0.000% | ∞ | 0.9% | 0.0% |
| A1 | humans | ; | 0.047% | 0.000% | ∞ | 2.0% | 0.0% |
| A1 | humans | = | 0.047% | 0.000% | ∞ | 1.4% | 0.0% |
| A1 | humans | Right | 0.027% | 0.000% | ∞ | 0.4% | 0.0% |
| A1 | humans | \ | 0.024% | 0.000% | ∞ | 0.9% | 0.0% |
| A1 | humans | Ctrl | 0.022% | 0.000% | ∞ | 0.6% | 0.0% |
| A1 | humans | ` | 0.014% | 0.000% | ∞ | 0.5% | 0.0% |
| A1 | humans | - (FF) | 0.009% | 0.000% | ∞ | 0.4% | 0.0% |
| A1 | humans | [ | 0.009% | 0.000% | ∞ | 0.4% | 0.0% |
| A1 | humans | keycode 97 | 0.009% | 0.000% | ∞ | 0.3% | 0.0% |
| A1 | humans | keycode 96 | 0.008% | 0.000% | ∞ | 0.3% | 0.0% |
| A1 | humans | keycode 59 | 0.006% | 0.000% | ∞ | 0.2% | 0.0% |
| A1 | humans | keycode 99 | 0.006% | 0.000% | ∞ | 0.3% | 0.0% |
| A1 | humans | Alt | 0.005% | 0.000% | ∞ | 0.2% | 0.0% |
| A1 | humans | keycode 105 | 0.004% | 0.000% | ∞ | 0.2% | 0.0% |
| A1 | humans | Insert | 0.004% | 0.000% | ∞ | 0.2% | 0.0% |
| A1 | humans | keycode 98 | 0.004% | 0.000% | ∞ | 0.2% | 0.0% |
| A1 | humans | keycode 102 | 0.004% | 0.000% | ∞ | 0.2% | 0.0% |
| A1 | humans | Delete | 0.003% | 0.000% | ∞ | 0.1% | 0.0% |
| A1 | humans | keycode 61 | 0.003% | 0.000% | ∞ | 0.2% | 0.0% |
| A1 | humans | End | 0.003% | 0.000% | ∞ | 0.1% | 0.0% |
| A1 | humans | keycode 100 | 0.003% | 0.000% | ∞ | 0.1% | 0.0% |
| A1 | humans | keycode 110 | 0.002% | 0.000% | ∞ | 0.1% | 0.0% |
| A2 | humans | Backspace | 5.435% | 0.000% | ∞ | 64.8% | 0.0% |
| A2 | humans | CapsLock | 0.590% | 0.000% | ∞ | 9.7% | 0.0% |
| A2 | humans | Left | 0.070% | 0.000% | ∞ | 0.9% | 0.0% |
| A2 | humans | ; | 0.047% | 0.000% | ∞ | 2.0% | 0.0% |
| A2 | humans | = | 0.047% | 0.000% | ∞ | 1.4% | 0.0% |
| A2 | humans | Right | 0.027% | 0.000% | ∞ | 0.4% | 0.0% |
| A2 | humans | \ | 0.024% | 0.000% | ∞ | 0.9% | 0.0% |
| A2 | humans | Ctrl | 0.022% | 0.000% | ∞ | 0.6% | 0.0% |
| A2 | humans | ` | 0.014% | 0.000% | ∞ | 0.5% | 0.0% |
| A2 | humans | - (FF) | 0.009% | 0.000% | ∞ | 0.4% | 0.0% |
| A2 | humans | [ | 0.009% | 0.000% | ∞ | 0.4% | 0.0% |
| A2 | humans | keycode 97 | 0.009% | 0.000% | ∞ | 0.3% | 0.0% |
| A2 | humans | keycode 96 | 0.008% | 0.000% | ∞ | 0.3% | 0.0% |
| A2 | humans | keycode 59 | 0.006% | 0.000% | ∞ | 0.2% | 0.0% |
| A2 | humans | keycode 99 | 0.006% | 0.000% | ∞ | 0.3% | 0.0% |
| A2 | humans | Alt | 0.005% | 0.000% | ∞ | 0.2% | 0.0% |
| A2 | humans | keycode 105 | 0.004% | 0.000% | ∞ | 0.2% | 0.0% |
| A2 | humans | Insert | 0.004% | 0.000% | ∞ | 0.2% | 0.0% |
| A2 | humans | keycode 98 | 0.004% | 0.000% | ∞ | 0.2% | 0.0% |
| A2 | humans | keycode 102 | 0.004% | 0.000% | ∞ | 0.2% | 0.0% |
| A2 | humans | Delete | 0.003% | 0.000% | ∞ | 0.1% | 0.0% |
| A2 | humans | keycode 61 | 0.003% | 0.000% | ∞ | 0.2% | 0.0% |
| A2 | humans | End | 0.003% | 0.000% | ∞ | 0.1% | 0.0% |
| A2 | humans | keycode 100 | 0.003% | 0.000% | ∞ | 0.1% | 0.0% |
| A2 | humans | keycode 110 | 0.002% | 0.000% | ∞ | 0.1% | 0.0% |
| A3 | humans | Backspace | 5.435% | 0.000% | ∞ | 64.8% | 0.0% |
| A3 | humans | CapsLock | 0.590% | 0.000% | ∞ | 9.7% | 0.0% |
| A3 | humans | Left | 0.070% | 0.000% | ∞ | 0.9% | 0.0% |
| A3 | humans | ; | 0.047% | 0.000% | ∞ | 2.0% | 0.0% |
| A3 | humans | = | 0.047% | 0.000% | ∞ | 1.4% | 0.0% |
| A3 | humans | Right | 0.027% | 0.000% | ∞ | 0.4% | 0.0% |
| A3 | humans | \ | 0.024% | 0.000% | ∞ | 0.9% | 0.0% |
| A3 | humans | Ctrl | 0.022% | 0.000% | ∞ | 0.6% | 0.0% |
| A3 | humans | ` | 0.014% | 0.000% | ∞ | 0.5% | 0.0% |
| A3 | humans | - (FF) | 0.009% | 0.000% | ∞ | 0.4% | 0.0% |
| A3 | humans | [ | 0.009% | 0.000% | ∞ | 0.4% | 0.0% |
| A3 | humans | keycode 97 | 0.009% | 0.000% | ∞ | 0.3% | 0.0% |
| A3 | humans | keycode 96 | 0.008% | 0.000% | ∞ | 0.3% | 0.0% |
| A3 | humans | keycode 59 | 0.006% | 0.000% | ∞ | 0.2% | 0.0% |
| A3 | humans | keycode 99 | 0.006% | 0.000% | ∞ | 0.3% | 0.0% |
| A3 | humans | Alt | 0.005% | 0.000% | ∞ | 0.2% | 0.0% |
| A3 | humans | keycode 105 | 0.004% | 0.000% | ∞ | 0.2% | 0.0% |
| A3 | humans | Insert | 0.004% | 0.000% | ∞ | 0.2% | 0.0% |
| A3 | humans | keycode 98 | 0.004% | 0.000% | ∞ | 0.2% | 0.0% |
| A3 | humans | keycode 102 | 0.004% | 0.000% | ∞ | 0.2% | 0.0% |
| A3 | humans | Delete | 0.003% | 0.000% | ∞ | 0.1% | 0.0% |
| A3 | humans | keycode 61 | 0.003% | 0.000% | ∞ | 0.2% | 0.0% |
| A3 | humans | End | 0.003% | 0.000% | ∞ | 0.1% | 0.0% |
| A3 | humans | keycode 100 | 0.003% | 0.000% | ∞ | 0.1% | 0.0% |
| A3 | humans | keycode 110 | 0.002% | 0.000% | ∞ | 0.1% | 0.0% |
| A4 | humans | CapsLock | 0.590% | 0.000% | ∞ | 9.7% | 0.0% |
| A4 | humans | Left | 0.070% | 0.000% | ∞ | 0.9% | 0.0% |
| A4 | humans | ; | 0.047% | 0.000% | ∞ | 2.0% | 0.0% |
| A4 | humans | = | 0.047% | 0.000% | ∞ | 1.4% | 0.0% |
| A4 | humans | Right | 0.027% | 0.000% | ∞ | 0.4% | 0.0% |
| A4 | humans | \ | 0.024% | 0.000% | ∞ | 0.9% | 0.0% |
| A4 | humans | Ctrl | 0.022% | 0.000% | ∞ | 0.6% | 0.0% |
| A4 | humans | ` | 0.014% | 0.000% | ∞ | 0.5% | 0.0% |
| A4 | humans | - (FF) | 0.009% | 0.000% | ∞ | 0.4% | 0.0% |
| A4 | humans | [ | 0.009% | 0.000% | ∞ | 0.4% | 0.0% |
| A4 | humans | keycode 97 | 0.009% | 0.000% | ∞ | 0.3% | 0.0% |
| A4 | humans | keycode 96 | 0.008% | 0.000% | ∞ | 0.3% | 0.0% |
| A4 | humans | keycode 59 | 0.006% | 0.000% | ∞ | 0.2% | 0.0% |
| A4 | humans | keycode 99 | 0.006% | 0.000% | ∞ | 0.3% | 0.0% |
| A4 | humans | Alt | 0.005% | 0.000% | ∞ | 0.2% | 0.0% |
| A4 | humans | keycode 105 | 0.004% | 0.000% | ∞ | 0.2% | 0.0% |
| A4 | humans | Insert | 0.004% | 0.000% | ∞ | 0.2% | 0.0% |
| A4 | humans | keycode 98 | 0.004% | 0.000% | ∞ | 0.2% | 0.0% |
| A4 | humans | keycode 102 | 0.004% | 0.000% | ∞ | 0.2% | 0.0% |
| A4 | humans | Delete | 0.003% | 0.000% | ∞ | 0.1% | 0.0% |
| A4 | humans | keycode 61 | 0.003% | 0.000% | ∞ | 0.2% | 0.0% |
| A4 | humans | End | 0.003% | 0.000% | ∞ | 0.1% | 0.0% |
| A4 | humans | keycode 100 | 0.003% | 0.000% | ∞ | 0.1% | 0.0% |
| A4 | humans | keycode 110 | 0.002% | 0.000% | ∞ | 0.1% | 0.0% |
| A5a | bots | Meta | 0.000% | 0.006% | ∞ | 0.0% | 0.3% |
| A5a | bots | PageUp | 0.000% | 0.002% | ∞ | 0.0% | 0.1% |
| A5a | bots | Home | 0.000% | 0.006% | 12.1 | 0.0% | 0.1% |
| A5a | humans | End | 0.003% | 0.000% | ∞ | 0.1% | 0.0% |
| A5b | humans | Backspace | 5.435% | 0.000% | ∞ | 64.8% | 0.0% |
| A5b | humans | CapsLock | 0.590% | 0.000% | ∞ | 9.7% | 0.0% |
| A5b | humans | Left | 0.070% | 0.000% | ∞ | 0.9% | 0.0% |
| A5b | humans | ; | 0.047% | 0.000% | ∞ | 2.0% | 0.0% |
| A5b | humans | = | 0.047% | 0.000% | ∞ | 1.4% | 0.0% |
| A5b | humans | Right | 0.027% | 0.000% | ∞ | 0.4% | 0.0% |
| A5b | humans | \ | 0.024% | 0.000% | ∞ | 0.9% | 0.0% |
| A5b | humans | Ctrl | 0.022% | 0.000% | ∞ | 0.6% | 0.0% |
| A5b | humans | ` | 0.014% | 0.000% | ∞ | 0.5% | 0.0% |
| A5b | humans | - (FF) | 0.009% | 0.000% | ∞ | 0.4% | 0.0% |
| A5b | humans | [ | 0.009% | 0.000% | ∞ | 0.4% | 0.0% |
| A5b | humans | keycode 97 | 0.009% | 0.000% | ∞ | 0.3% | 0.0% |
| A5b | humans | keycode 96 | 0.008% | 0.000% | ∞ | 0.3% | 0.0% |
| A5b | humans | keycode 59 | 0.006% | 0.000% | ∞ | 0.2% | 0.0% |
| A5b | humans | keycode 99 | 0.006% | 0.000% | ∞ | 0.3% | 0.0% |
| A5b | humans | Alt | 0.005% | 0.000% | ∞ | 0.2% | 0.0% |
| A5b | humans | keycode 105 | 0.004% | 0.000% | ∞ | 0.2% | 0.0% |
| A5b | humans | Insert | 0.004% | 0.000% | ∞ | 0.2% | 0.0% |
| A5b | humans | keycode 98 | 0.004% | 0.000% | ∞ | 0.2% | 0.0% |
| A5b | humans | keycode 102 | 0.004% | 0.000% | ∞ | 0.2% | 0.0% |
| A5b | humans | Delete | 0.003% | 0.000% | ∞ | 0.1% | 0.0% |
| A5b | humans | keycode 61 | 0.003% | 0.000% | ∞ | 0.2% | 0.0% |
| A5b | humans | End | 0.003% | 0.000% | ∞ | 0.1% | 0.0% |
| A5b | humans | keycode 100 | 0.003% | 0.000% | ∞ | 0.1% | 0.0% |
| A5b | humans | keycode 110 | 0.002% | 0.000% | ∞ | 0.1% | 0.0% |

**Keycode bigrams ≥ 10× more frequent in one group** (same rule): 1111 bigrams in total (`realism_enriched_bigrams.csv` lists all). Counts per attacker and direction:

| attacker | more_in | n |
|---|---|---|
| A1 | bots | 1 |
| A1 | humans | 216 |
| A2 | bots | 1 |
| A2 | humans | 216 |
| A3 | bots | 1 |
| A3 | humans | 216 |
| A4 | bots | 86 |
| A4 | humans | 131 |
| A5a | bots | 14 |
| A5a | humans | 16 |
| A5b | bots | 1 |
| A5b | humans | 212 |

The 15 most frequent (in the group where they are enriched) per attacker and direction:

| attacker | more_in | name | freq_humans | freq_bots | ratio | sessions_humans | sessions_bots |
|---|---|---|---|---|---|---|---|
| A1 | bots | Q -> Shift | 0.000% | 0.003% | ∞ | 0.0% | 0.1% |
| A1 | humans | Backspace -> Backspace | 2.529% | 0.000% | ∞ | 41.1% | 0.0% |
| A1 | humans | Space -> Backspace | 0.444% | 0.000% | ∞ | 16.9% | 0.0% |
| A1 | humans | E -> Backspace | 0.270% | 0.000% | ∞ | 11.3% | 0.0% |
| A1 | humans | Backspace -> E | 0.238% | 0.000% | ∞ | 10.2% | 0.0% |
| A1 | humans | Backspace -> Shift | 0.216% | 0.000% | ∞ | 8.9% | 0.0% |
| A1 | humans | Backspace -> T | 0.188% | 0.000% | ∞ | 8.3% | 0.0% |
| A1 | humans | I -> Backspace | 0.188% | 0.000% | ∞ | 8.1% | 0.0% |
| A1 | humans | A -> Backspace | 0.174% | 0.000% | ∞ | 7.7% | 0.0% |
| A1 | humans | S -> Backspace | 0.168% | 0.000% | ∞ | 7.4% | 0.0% |
| A1 | humans | Backspace -> I | 0.166% | 0.000% | ∞ | 7.2% | 0.0% |
| A1 | humans | T -> Backspace | 0.163% | 0.000% | ∞ | 7.3% | 0.0% |
| A1 | humans | Backspace -> Space | 0.153% | 0.000% | ∞ | 6.7% | 0.0% |
| A1 | humans | Backspace -> O | 0.151% | 0.000% | ∞ | 6.8% | 0.0% |
| A1 | humans | Backspace -> R | 0.151% | 0.000% | ∞ | 6.6% | 0.0% |
| A1 | humans | Backspace -> A | 0.146% | 0.000% | ∞ | 6.4% | 0.0% |
| A2 | bots | Q -> Shift | 0.000% | 0.003% | ∞ | 0.0% | 0.1% |
| A2 | humans | Backspace -> Backspace | 2.529% | 0.000% | ∞ | 41.1% | 0.0% |
| A2 | humans | Space -> Backspace | 0.444% | 0.000% | ∞ | 16.9% | 0.0% |
| A2 | humans | E -> Backspace | 0.270% | 0.000% | ∞ | 11.3% | 0.0% |
| A2 | humans | Backspace -> E | 0.238% | 0.000% | ∞ | 10.2% | 0.0% |
| A2 | humans | Backspace -> Shift | 0.216% | 0.000% | ∞ | 8.9% | 0.0% |
| A2 | humans | I -> Backspace | 0.188% | 0.000% | ∞ | 8.1% | 0.0% |
| A2 | humans | Backspace -> T | 0.188% | 0.000% | ∞ | 8.3% | 0.0% |
| A2 | humans | A -> Backspace | 0.174% | 0.000% | ∞ | 7.7% | 0.0% |
| A2 | humans | S -> Backspace | 0.168% | 0.000% | ∞ | 7.4% | 0.0% |
| A2 | humans | Backspace -> I | 0.166% | 0.000% | ∞ | 7.2% | 0.0% |
| A2 | humans | T -> Backspace | 0.163% | 0.000% | ∞ | 7.3% | 0.0% |
| A2 | humans | Backspace -> Space | 0.153% | 0.000% | ∞ | 6.7% | 0.0% |
| A2 | humans | Backspace -> O | 0.151% | 0.000% | ∞ | 6.8% | 0.0% |
| A2 | humans | Backspace -> R | 0.151% | 0.000% | ∞ | 6.6% | 0.0% |
| A2 | humans | Backspace -> A | 0.146% | 0.000% | ∞ | 6.4% | 0.0% |
| A3 | bots | Q -> Shift | 0.000% | 0.003% | ∞ | 0.0% | 0.1% |
| A3 | humans | Backspace -> Backspace | 2.529% | 0.000% | ∞ | 41.1% | 0.0% |
| A3 | humans | Space -> Backspace | 0.444% | 0.000% | ∞ | 16.9% | 0.0% |
| A3 | humans | E -> Backspace | 0.270% | 0.000% | ∞ | 11.3% | 0.0% |
| A3 | humans | Backspace -> E | 0.238% | 0.000% | ∞ | 10.2% | 0.0% |
| A3 | humans | Backspace -> Shift | 0.216% | 0.000% | ∞ | 8.9% | 0.0% |
| A3 | humans | Backspace -> T | 0.188% | 0.000% | ∞ | 8.3% | 0.0% |
| A3 | humans | I -> Backspace | 0.188% | 0.000% | ∞ | 8.1% | 0.0% |
| A3 | humans | A -> Backspace | 0.174% | 0.000% | ∞ | 7.7% | 0.0% |
| A3 | humans | S -> Backspace | 0.168% | 0.000% | ∞ | 7.4% | 0.0% |
| A3 | humans | Backspace -> I | 0.166% | 0.000% | ∞ | 7.2% | 0.0% |
| A3 | humans | T -> Backspace | 0.163% | 0.000% | ∞ | 7.3% | 0.0% |
| A3 | humans | Backspace -> Space | 0.153% | 0.000% | ∞ | 6.7% | 0.0% |
| A3 | humans | Backspace -> O | 0.151% | 0.000% | ∞ | 6.8% | 0.0% |
| A3 | humans | Backspace -> R | 0.151% | 0.000% | ∞ | 6.6% | 0.0% |
| A3 | humans | Backspace -> A | 0.146% | 0.000% | ∞ | 6.4% | 0.0% |
| A4 | bots | Q -> Backspace | 0.003% | 0.050% | 14.6 | 0.2% | 2.3% |
| A4 | bots | Space -> Z | 0.001% | 0.030% | 31.5 | 0.0% | 1.4% |
| A4 | bots | H -> D | 0.002% | 0.025% | 12.7 | 0.1% | 1.2% |
| A4 | bots | B -> D | 0.001% | 0.018% | 18.8 | 0.0% | 0.8% |
| A4 | bots | Space -> X | 0.001% | 0.017% | 11.8 | 0.1% | 0.8% |
| A4 | bots | Y -> H | 0.001% | 0.017% | 11.8 | 0.1% | 0.8% |
| A4 | bots | F -> H | 0.000% | 0.015% | 31.5 | 0.0% | 0.7% |
| A4 | bots | T -> D | 0.000% | 0.013% | 27.4 | 0.0% | 0.6% |
| A4 | bots | W -> D | 0.000% | 0.013% | ∞ | 0.0% | 0.6% |
| A4 | bots | M -> D | 0.001% | 0.013% | 13.2 | 0.0% | 0.6% |
| A4 | bots | H -> K | 0.000% | 0.012% | ∞ | 0.0% | 0.6% |
| A4 | bots | E -> Z | 0.001% | 0.012% | 12.7 | 0.0% | 0.6% |
| A4 | bots | K -> R | 0.001% | 0.012% | 12.2 | 0.0% | 0.6% |
| A4 | bots | Z -> Space | 0.001% | 0.011% | 11.7 | 0.0% | 0.5% |
| A4 | bots | W -> K | 0.000% | 0.011% | 23.4 | 0.0% | 0.5% |
| A4 | humans | Backspace -> Space | 0.153% | 0.000% | ∞ | 6.7% | 0.0% |
| A4 | humans | . -> Space | 0.127% | 0.003% | 37.0 | 6.0% | 0.2% |
| A4 | humans | Space -> CapsLock | 0.103% | 0.000% | ∞ | 3.5% | 0.0% |
| A4 | humans | Backspace -> . | 0.098% | 0.000% | ∞ | 4.6% | 0.0% |
| A4 | humans | CapsLock -> I | 0.067% | 0.000% | ∞ | 3.0% | 0.0% |
| A4 | humans | CapsLock -> H | 0.056% | 0.000% | ∞ | 2.6% | 0.0% |
| A4 | humans | , -> Backspace | 0.052% | 0.004% | 11.8 | 2.3% | 0.2% |
| A4 | humans | CapsLock -> A | 0.052% | 0.000% | ∞ | 2.1% | 0.0% |
| A4 | humans | Left -> Left | 0.051% | 0.000% | ∞ | 0.5% | 0.0% |
| A4 | humans | Shift -> Shift | 0.050% | 0.000% | ∞ | 2.1% | 0.0% |
| A4 | humans | I -> CapsLock | 0.050% | 0.000% | ∞ | 2.3% | 0.0% |
| A4 | humans | CapsLock -> T | 0.048% | 0.000% | ∞ | 2.1% | 0.0% |
| A4 | humans | Backspace -> ' | 0.047% | 0.000% | ∞ | 2.2% | 0.0% |
| A4 | humans | Backspace -> , | 0.046% | 0.000% | ∞ | 2.0% | 0.0% |
| A4 | humans | CapsLock -> O | 0.044% | 0.000% | ∞ | 1.9% | 0.0% |
| A5a | bots | C -> D | 0.000% | 0.005% | ∞ | 0.0% | 0.3% |
| A5a | bots | Home -> Home | 0.000% | 0.003% | ∞ | 0.0% | 0.1% |
| A5a | bots | Z -> H | 0.000% | 0.003% | ∞ | 0.0% | 0.1% |
| A5a | bots | digit 0 -> , | 0.000% | 0.003% | ∞ | 0.0% | 0.1% |
| A5a | bots | S -> PageUp | 0.000% | 0.002% | ∞ | 0.0% | 0.1% |
| A5a | bots | R -> keycode 61 | 0.000% | 0.002% | ∞ | 0.0% | 0.1% |
| A5a | bots | A -> - (FF) | 0.000% | 0.002% | ∞ | 0.0% | 0.1% |
| A5a | bots | digit 5 -> Backspace | 0.000% | 0.002% | ∞ | 0.0% | 0.1% |
| A5a | bots | V -> digit 4 | 0.000% | 0.002% | ∞ | 0.0% | 0.1% |
| A5a | bots | T -> Meta | 0.000% | 0.002% | ∞ | 0.0% | 0.1% |
| A5a | bots | W -> D | 0.000% | 0.002% | ∞ | 0.0% | 0.1% |
| A5a | bots | ` -> Space | 0.000% | 0.002% | ∞ | 0.0% | 0.1% |
| A5a | bots | Meta -> Shift | 0.000% | 0.002% | ∞ | 0.0% | 0.1% |
| A5a | bots | PageUp -> Shift | 0.000% | 0.002% | ∞ | 0.0% | 0.1% |
| A5a | humans | = -> = | 0.014% | 0.000% | 27.9 | 0.3% | 0.0% |
| A5a | humans | S -> F | 0.006% | 0.000% | ∞ | 0.3% | 0.0% |
| A5a | humans | R -> CapsLock | 0.006% | 0.000% | ∞ | 0.3% | 0.0% |
| A5a | humans | Backspace -> Z | 0.005% | 0.000% | ∞ | 0.2% | 0.0% |
| A5a | humans | Y -> / | 0.005% | 0.000% | ∞ | 0.2% | 0.0% |
| A5a | humans | T -> = | 0.004% | 0.000% | ∞ | 0.2% | 0.0% |
| A5a | humans | . -> ' | 0.004% | 0.000% | ∞ | 0.2% | 0.0% |
| A5a | humans | digit 7 -> . | 0.003% | 0.000% | ∞ | 0.1% | 0.0% |
| A5a | humans | digit 6 -> . | 0.003% | 0.000% | ∞ | 0.1% | 0.0% |
| A5a | humans | ' -> Y | 0.002% | 0.000% | ∞ | 0.1% | 0.0% |
| A5a | humans | Ctrl -> Backspace | 0.002% | 0.000% | ∞ | 0.1% | 0.0% |
| A5a | humans | digit 3 -> digit 5 | 0.002% | 0.000% | ∞ | 0.1% | 0.0% |
| A5a | humans | M -> / | 0.002% | 0.000% | ∞ | 0.1% | 0.0% |
| A5a | humans | Space -> Left | 0.002% | 0.000% | ∞ | 0.1% | 0.0% |
| A5a | humans | Shift -> CapsLock | 0.002% | 0.000% | ∞ | 0.1% | 0.0% |
| A5b | bots | Q -> Shift | 0.000% | 0.003% | ∞ | 0.0% | 0.1% |
| A5b | humans | Backspace -> Backspace | 2.529% | 0.000% | ∞ | 41.1% | 0.0% |
| A5b | humans | Space -> Backspace | 0.444% | 0.000% | ∞ | 16.9% | 0.0% |
| A5b | humans | E -> Backspace | 0.270% | 0.000% | ∞ | 11.3% | 0.0% |
| A5b | humans | Backspace -> E | 0.238% | 0.000% | ∞ | 10.2% | 0.0% |
| A5b | humans | Backspace -> Shift | 0.216% | 0.000% | ∞ | 8.9% | 0.0% |
| A5b | humans | I -> Backspace | 0.188% | 0.000% | ∞ | 8.1% | 0.0% |
| A5b | humans | Backspace -> T | 0.188% | 0.000% | ∞ | 8.3% | 0.0% |
| A5b | humans | A -> Backspace | 0.174% | 0.000% | ∞ | 7.7% | 0.0% |
| A5b | humans | S -> Backspace | 0.168% | 0.000% | ∞ | 7.4% | 0.0% |
| A5b | humans | Backspace -> I | 0.166% | 0.000% | ∞ | 7.2% | 0.0% |
| A5b | humans | T -> Backspace | 0.163% | 0.000% | ∞ | 7.3% | 0.0% |
| A5b | humans | Backspace -> Space | 0.153% | 0.000% | ∞ | 6.7% | 0.0% |
| A5b | humans | Backspace -> R | 0.151% | 0.000% | ∞ | 6.6% | 0.0% |
| A5b | humans | Backspace -> O | 0.151% | 0.000% | ∞ | 6.8% | 0.0% |
| A5b | humans | Backspace -> A | 0.146% | 0.000% | ∞ | 6.4% | 0.0% |

## 2. Swap test (TEST split, frozen weights, frozen τ10)

Each test bot is paired with the test human it was generated for (same session, so the same target sentence). S1 = human keys + human timing (all test humans), S2 = bot keys + bot timing (the original attack), S3 = human keys + bot timing, S4 = bot keys + human timing; S3/S4 are aligned by position and truncated to the shorter length. Controls (not in the spec): S1t / S2t = human / bot truncated to the same pair length.

**Length mismatch and alignment:**

| attacker | pairs | length mismatch | bot longer / shorter | |diff| mean / median / p95 / max (mismatched) | keys dropped human / bot | aligned positions with the same key | identical key sequences |
|---|---|---|---|---|---|---|---|
| A1 | 4,344 | 3,349 (77.1%) | 269 / 3,080 | 7.2 / 5 / 22 / 83 | 11.2% / 0.2% | 48.4% | 864 |
| A2 | 4,344 | 3,349 (77.1%) | 269 / 3,080 | 7.2 / 5 / 22 / 83 | 11.2% / 0.2% | 48.4% | 864 |
| A3 | 4,344 | 3,349 (77.1%) | 269 / 3,080 | 7.2 / 5 / 22 / 83 | 11.2% / 0.2% | 48.4% | 864 |
| A4 | 4,344 | 3,745 (86.2%) | 1,687 / 2,058 | 7.6 / 5 / 21 / 83 | 7.4% / 6.1% | 35.5% | 315 |
| A5a | 4,344 | 4,286 (98.7%) | 2,133 / 2,153 | 21.6 / 18 / 52 / 167 | 22.1% / 21.8% | 8.7% | 0 |
| A5b | 4,344 | 3,349 (77.1%) | 269 / 3,080 | 7.2 / 5 / 22 / 83 | 11.2% / 0.2% | 48.4% | 853 |

**Caveat (found in the quick run, before the full run):** positional alignment pairs each timing with the key at the same POSITION. After the first place where the human corrected a typo (or the replayed human typed differently), the positions no longer carry the same character. The 'aligned positions with the same key' column shows how often they do. S3 and S4 therefore combine the keys of one sequence with timings that belong to OTHER keys. A detector that models timing *given the key* will flag that combination whoever made the keys. A5a (a verbatim replay of another human typing the same sentence) is the built-in control: its S3 and S4 are human keys + human timing from two different people.

**D4-primary**: APCER@τ10 (attacks accepted) and AUC vs S1; mean ± sd over seeds 42/43/44:

| attacker | S1 BPCER@τ10 | S2 APCER | S3 APCER | S4 APCER | S2t APCER | S1t APCER | S2 AUC | S3 AUC | S4 AUC | S2t AUC | S1t AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 | 11.2% ± 0.1 | 71.9% ± 1.8 | 45.3% ± 1.8 | 45.0% ± 0.5 | 72.0% ± 1.8 | 88.6% ± 0.1 | 0.799 ± 0.006 | 0.867 ± 0.005 | 0.756 ± 0.002 | 0.798 ± 0.006 | 0.496 ± 0.000 |
| A2 | 11.2% ± 0.1 | 1.1% ± 0.1 | 0.5% ± 0.0 | 45.0% ± 0.5 | 1.1% ± 0.1 | 88.6% ± 0.1 | 0.987 ± 0.001 | 0.994 ± 0.000 | 0.756 ± 0.002 | 0.987 ± 0.001 | 0.496 ± 0.000 |
| A3 | 11.2% ± 0.1 | 5.0% ± 0.6 | 2.0% ± 0.2 | 45.0% ± 0.5 | 5.1% ± 0.6 | 88.6% ± 0.1 | 0.968 ± 0.002 | 0.984 ± 0.001 | 0.756 ± 0.002 | 0.968 ± 0.002 | 0.496 ± 0.000 |
| A4 | 11.2% ± 0.1 | 2.8% ± 0.3 | 0.8% ± 0.1 | 30.3% ± 0.4 | 3.1% ± 0.3 | 88.5% ± 0.1 | 0.979 ± 0.001 | 0.992 ± 0.000 | 0.849 ± 0.002 | 0.978 ± 0.001 | 0.497 ± 0.000 |
| A5a | 11.2% ± 0.1 | 90.3% ± 0.2 | 13.6% ± 0.5 | 11.1% ± 0.7 | 89.1% ± 0.4 | 87.6% ± 0.1 | 0.479 ± 0.002 | 0.947 ± 0.001 | 0.955 ± 0.002 | 0.484 ± 0.002 | 0.503 ± 0.000 |
| A5b | 11.2% ± 0.1 | 13.6% ± 0.4 | 10.0% ± 0.4 | 45.0% ± 0.5 | 13.8% ± 0.4 | 88.6% ± 0.1 | 0.945 ± 0.002 | 0.958 ± 0.001 | 0.756 ± 0.002 | 0.945 ± 0.002 | 0.496 ± 0.000 |

**D5 (E5, own attacker)**: APCER@τ10 (attacks accepted) and AUC vs S1; mean ± sd over seeds 42/43/44 (each D5 model tested only on the attacker it was trained on):

| attacker | S1 BPCER@τ10 | S2 APCER | S3 APCER | S4 APCER | S2t APCER | S1t APCER | S2 AUC | S3 AUC | S4 AUC | S2t AUC | S1t AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 | 8.6% ± 0.4 | 0.0% ± 0.0 | 24.4% ± 10.5 | 69.4% ± 2.0 | 0.0% ± 0.0 | 90.8% ± 0.2 | 1.000 ± 0.000 | 0.927 ± 0.030 | 0.733 ± 0.012 | 1.000 ± 0.000 | 0.525 ± 0.005 |
| A2 | 8.6% ± 0.6 | 0.0% ± 0.0 | 46.6% ± 4.0 | 47.5% ± 2.2 | 0.0% ± 0.0 | 90.6% ± 0.8 | 0.999 ± 0.000 | 0.779 ± 0.031 | 0.873 ± 0.014 | 0.998 ± 0.000 | 0.509 ± 0.006 |
| A3 | 7.8% ± 0.1 | 0.1% ± 0.0 | 61.4% ± 0.7 | 49.0% ± 1.0 | 0.2% ± 0.0 | 91.5% ± 0.2 | 0.998 ± 0.000 | 0.702 ± 0.017 | 0.890 ± 0.007 | 0.998 ± 0.000 | 0.512 ± 0.005 |
| A4 | 8.1% ± 0.1 | 0.7% ± 0.1 | 8.4% ± 1.4 | 54.9% ± 4.1 | 2.8% ± 0.8 | 93.1% ± 0.1 | 0.996 ± 0.001 | 0.967 ± 0.006 | 0.785 ± 0.024 | 0.990 ± 0.002 | 0.481 ± 0.007 |
| A5b | 8.8% ± 0.4 | 0.2% ± 0.1 | 57.3% ± 5.8 | 24.1% ± 0.5 | 0.4% ± 0.2 | 90.3% ± 0.4 | 0.997 ± 0.000 | 0.759 ± 0.038 | 0.942 ± 0.003 | 0.997 ± 0.000 | 0.492 ± 0.005 |

**Share of the S2 effect reproduced by each swap.** Effect(S) = AUC(S1 vs S) − 0.5 (and, second, flag rate(S) − flag rate(S1)); share = Effect(S3) / Effect(S2) (timing) and Effect(S4) / Effect(S2) (keys), per seed, then mean ± sd. Undefined when |Effect(S2)| < 0.05. The shares need not add to 1 (interaction, misalignment). Label rule, fixed before the full run: *signal* if the timing share ≥ 0.75 and the key share ≤ 0.25; *artefact* if the key share ≥ 0.75 and the timing share ≤ 0.25; *mixed* otherwise.

| detector | attacker | Effect(S2) AUC | timing share (S3) AUC | key share (S4) AUC | timing share (S3) flag | key share (S4) flag | truncation control (S2t) AUC | label (AUC shares) |
|---|---|---|---|---|---|---|---|---|
| D4-primary | A1 | 0.30 ± 0.01 | 1.23 ± 0.01 | 0.86 ± 0.01 | 2.58 ± 0.18 | 2.60 ± 0.24 | 1.00 ± 0.00 | mixed |
| D4-primary | A2 | 0.49 ± 0.00 | 1.01 ± 0.00 | 0.53 ± 0.00 | 1.01 ± 0.00 | 0.50 ± 0.00 | 1.00 ± 0.00 | mixed |
| D4-primary | A3 | 0.47 ± 0.00 | 1.03 ± 0.00 | 0.55 ± 0.00 | 1.04 ± 0.00 | 0.52 ± 0.00 | 1.00 ± 0.00 | mixed |
| D4-primary | A4 | 0.48 ± 0.00 | 1.03 ± 0.00 | 0.73 ± 0.00 | 1.02 ± 0.00 | 0.68 ± 0.00 | 1.00 ± 0.00 | mixed |
| D4-primary | A5a | -0.02 ± 0.00 | — | — | — | — | — | n/a (S2 not detected) |
| D4-primary | A5b | 0.45 ± 0.00 | 1.03 ± 0.00 | 0.58 ± 0.00 | 1.05 ± 0.01 | 0.58 ± 0.00 | 1.00 ± 0.00 | mixed |
| D5 (E5, own attacker) | A1 | 0.50 ± 0.00 | 0.85 ± 0.06 | 0.47 ± 0.02 | 0.73 ± 0.11 | 0.24 ± 0.02 | 1.00 ± 0.00 | mixed |
| D5 (E5, own attacker) | A2 | 0.50 ± 0.00 | 0.56 ± 0.06 | 0.75 ± 0.03 | 0.49 ± 0.05 | 0.48 ± 0.02 | 1.00 ± 0.00 | mixed |
| D5 (E5, own attacker) | A3 | 0.50 ± 0.00 | 0.41 ± 0.03 | 0.78 ± 0.01 | 0.34 ± 0.01 | 0.47 ± 0.01 | 1.00 ± 0.00 | mixed |
| D5 (E5, own attacker) | A4 | 0.50 ± 0.00 | 0.94 ± 0.01 | 0.57 ± 0.05 | 0.92 ± 0.01 | 0.41 ± 0.04 | 0.99 ± 0.00 | mixed |
| D5 (E5, own attacker) | A5b | 0.50 ± 0.00 | 0.52 ± 0.08 | 0.89 ± 0.01 | 0.37 ± 0.06 | 0.74 ± 0.00 | 1.00 ± 0.00 | mixed |

**Identical-key-sequence pairs (addition, not in the spec).** These are the pairs where the human typed exactly the bot's keycode sequence: same length, same key at every position (so no correction, the same Shift use). Here the keys are identical by construction, so any separation of the bot from its human can only come from timing. S1 and S2 are restricted to these pairs:

| detector | attacker | identical pairs | humans BPCER@τ10 | bots APCER@τ10 | AUC |
|---|---|---|---|---|---|
| D4-primary | A1 | 864 | 8.0% ± 0.3 | 66.3% ± 1.6 | 0.863 ± 0.002 |
| D4-primary | A2 | 864 | 8.0% ± 0.3 | 1.8% ± 0.1 | 0.990 ± 0.001 |
| D4-primary | A3 | 864 | 8.0% ± 0.3 | 6.9% ± 0.6 | 0.973 ± 0.002 |
| D4-primary | A4 | 315 | 8.1% ± 0.5 | 4.6% ± 0.8 | 0.978 ± 0.002 |
| D4-primary | A5a | 0 | — | — | — |
| D4-primary | A5b | 853 | 7.9% ± 0.3 | 16.8% ± 0.7 | 0.952 ± 0.001 |
| D5 (E5, own attacker) | A1 | 864 | 21.1% ± 0.8 | 0.0% ± 0.0 | 1.000 ± 0.000 |
| D5 (E5, own attacker) | A2 | 864 | 27.2% ± 2.2 | 0.0% ± 0.1 | 0.993 ± 0.001 |
| D5 (E5, own attacker) | A3 | 864 | 26.7% ± 0.7 | 0.2% ± 0.1 | 0.992 ± 0.001 |
| D5 (E5, own attacker) | A4 | 315 | 16.6% ± 0.5 | 1.6% ± 0.3 | 0.987 ± 0.002 |
| D5 (E5, own attacker) | A5b | 853 | 31.4% ± 1.7 | 0.1% ± 0.1 | 0.988 ± 0.001 |

## 3. Letters-only D4-primary

Session score = mean per-step NLL over the steps where both the current and the next key are a–z or space (all other steps masked; the recurrent state still sees the whole history). τ10 recomputed on val humans with the same masked score (10% of val humans rejected). Test humans: BPCER@τ10 11.1% ± 0.3; letter steps are 80.6% of human steps; sessions without any letter step: test humans 0, val humans 0 (excluded).

| attacker | APCER@τ10 original | APCER@τ10 letters-only | AUC original | AUC letters-only | letter-step share | sessions without letter steps |
|---|---|---|---|---|---|---|
| A1 | 71.9% ± 1.8 | 70.2% ± 2.2 | 0.799 ± 0.006 | 0.805 ± 0.009 | 88.3% | 0 |
| A2 | 1.1% ± 0.1 | 1.5% ± 0.1 | 0.987 ± 0.001 | 0.983 ± 0.001 | 88.3% | 0 |
| A3 | 5.0% ± 0.6 | 5.8% ± 0.7 | 0.968 ± 0.002 | 0.961 ± 0.001 | 88.3% | 0 |
| A4 | 2.8% ± 0.3 | 4.3% ± 0.4 | 0.979 ± 0.001 | 0.970 ± 0.001 | 81.6% | 0 |
| A5a | 90.3% ± 0.2 | 90.3% ± 0.3 | 0.479 ± 0.002 | 0.474 ± 0.001 | 80.5% | 3 |
| A5b | 13.6% ± 0.4 | 15.3% ± 0.6 | 0.945 ± 0.002 | 0.938 ± 0.001 | 88.3% | 0 |

## 4. Verdict

_Verdict not written yet (see `verdict.md`)._

## Files

`realism_*.csv`, `swap_length_mismatch.csv`, `swap_per_seed.csv`, `swap_shares_per_seed.csv`, `identical_keys_per_seed.csv`, `letters_only_per_seed.csv`, `reproduction_check.csv`, `phase3c_meta.json`; code `src/kl/phase3c/`, `scripts/40_run_phase3c.py`, `scripts/41_report_phase3c.py`; spec `docs/PHASE3C_SPEC.md`.
