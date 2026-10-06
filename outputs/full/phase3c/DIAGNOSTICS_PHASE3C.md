# Phase 3c — artefact check of D4 / D5 (**POST-HOC**, diagnostic only)

_Run mode **full**. **POST-HOC**: designed after the Phase 3b results were seen; nothing here was pre-registered. No model, attacker, threshold or split was changed and nothing was retrained. The TEST split is re-used post hoc with the frozen Phase 3b weights and the frozen τ10 (letters-only: τ10 recomputed on val humans for the new masked score). Freeze guards (Phases 1, 2, 3a, 3b pre-registration and `docs/phase3b_frozen_hashes.json`) pass before and after the run. Re-scoring S1/S2 reproduces the stored Phase 3b scores: max |diff| = 0. Device cuda (NVIDIA GeForce RTX 3050 6GB Laptop GPU)._

Question: are D4's and D5's detections of A2–A5b driven by timing given the keys (a real signal), or by differences in the key sequences the bots generate (an implementation artefact)?

## 1. Keycode realism (test humans vs test bots, raw sequences before truncation)

| group | sessions | length mean / median [p5, p95] | UNK % | Backspace / 100 keys | Delete / 100 keys | sessions with Bksp/Del | CapsLock / 100 keys | non-letter/space share |
|---|---|---|---|---|---|---|---|---|
| humans | 43,248 | 48.4 / 47 [22, 79] | 0.002 | 5.28 | 0.002 | 62.7% | 0.51 | 12.4% |
| A1 | 43,248 | 43.2 / 42 [21, 69] | 0.000 | 0.00 | 0.000 | 0.0% | 0.00 | 7.2% |
| A2 | 43,248 | 43.2 / 42 [21, 69] | 0.000 | 0.00 | 0.000 | 0.0% | 0.00 | 7.2% |
| A3 | 43,248 | 43.2 / 42 [21, 69] | 0.000 | 0.00 | 0.000 | 0.0% | 0.00 | 7.2% |
| A4 | 43,248 | 47.7 / 46 [22, 78] | 0.000 | 4.65 | 0.000 | 55.6% | 0.00 | 11.4% |
| A5a | 43,248 | 48.6 / 47 [22, 80] | 0.002 | 5.30 | 0.003 | 63.0% | 0.58 | 12.4% |
| A5b | 43,248 | 43.2 / 42 [21, 69] | 0.000 | 0.00 | 0.000 | 0.0% | 0.00 | 7.2% |

**JSD** (Jensen–Shannon divergence, base 2, 0 = identical, 1 = disjoint) of the keycode and keycode-bigram distributions, test humans vs each attacker:

| attacker | jsd_keycode | jsd_bigram |
|---|---|---|
| A1 | 0.0315 | 0.053 |
| A2 | 0.0315 | 0.053 |
| A3 | 0.0315 | 0.053 |
| A4 | 0.0055 | 0.019 |
| A5a | 0.0001 | 0.0011 |
| A5b | 0.0315 | 0.0527 |

**Shift pattern.** Capitals are counted in the target sentence (both sides type the same sentence); 'Shift-char' also counts punctuation that needs Shift. Position = the next non-Shift key after each Shift press.

| group | Shift per capital | Shift per Shift-char | next key a letter | Shift released before next press | released during next key | released after next key | Shift→key lead (median ms) | Shift hold median [p10, p90] ms |
|---|---|---|---|---|---|---|---|---|
| humans | 1.023 | 0.954 | 90.4% | 5.5% | 63.1% | 31.3% | 160 | 248 [143, 567] |
| A1 | 1.072 | 1.000 | 93.3% | 0.0% | 0.0% | 100.0% | 239 | 384 [241, 531] |
| A2 | 1.072 | 1.000 | 93.3% | 0.0% | 78.8% | 21.2% | 145 | 232 [126, 466] |
| A3 | 1.072 | 1.000 | 93.3% | 0.0% | 78.5% | 21.5% | 144 | 230 [123, 453] |
| A4 | 1.121 | 1.045 | 93.4% | 0.0% | 74.9% | 25.1% | 145 | 237 [127, 495] |
| A5a | 1.011 | 0.943 | 90.4% | 5.7% | 63.7% | 30.6% | 160 | 246 [140, 567] |
| A5b | 1.072 | 1.000 | 93.2% | 38.8% | 46.9% | 14.3% | 169 | 159 [79, 429] |

**Keys other than letters and space** (punctuation, digits, control keys), per 1,000 keystrokes; keycodes present in ≥ 0.1% of the sessions of at least one group:

| keycode | name | humans | A1 | A2 | A3 | A4 | A5a | A5b |
|---|---|---|---|---|---|---|---|---|
| 8 | Backspace | 52.80 | 0.00 | 0.00 | 0.00 | 46.49 | 52.97 | 0.00 |
| 16 | Shift | 33.01 | 38.75 | 38.75 | 38.75 | 36.67 | 32.51 | 38.75 |
| 190 | . | 17.69 | 20.62 | 20.62 | 20.62 | 19.18 | 17.56 | 20.62 |
| 20 | CapsLock | 5.08 | 0.00 | 0.00 | 0.00 | 0.00 | 5.81 | 0.00 |
| 222 | ' | 3.71 | 4.22 | 4.22 | 4.22 | 3.93 | 3.75 | 4.22 |
| 188 | , | 3.05 | 2.70 | 2.70 | 2.70 | 2.51 | 3.18 | 2.70 |
| 191 | / | 2.47 | 2.35 | 2.35 | 2.35 | 2.18 | 2.27 | 2.35 |
| 49 | digit 1 | 0.70 | 0.86 | 0.86 | 0.86 | 0.79 | 0.70 | 0.86 |
| 37 | Left | 0.65 | 0.00 | 0.00 | 0.00 | 0.00 | 0.63 | 0.00 |
| 189 | - | 0.53 | 0.62 | 0.62 | 0.62 | 0.57 | 0.44 | 0.62 |
| 186 | ; | 0.49 | 0.00 | 0.00 | 0.00 | 0.00 | 0.41 | 0.00 |
| 48 | digit 0 | 0.46 | 0.53 | 0.53 | 0.53 | 0.48 | 0.45 | 0.53 |
| 39 | Right | 0.39 | 0.00 | 0.00 | 0.00 | 0.00 | 0.24 | 0.00 |
| 187 | = | 0.37 | 0.00 | 0.00 | 0.00 | 0.00 | 0.37 | 0.00 |
| 50 | digit 2 | 0.31 | 0.35 | 0.35 | 0.35 | 0.32 | 0.29 | 0.35 |
| 51 | digit 3 | 0.19 | 0.22 | 0.22 | 0.22 | 0.21 | 0.18 | 0.22 |
| 17 | Ctrl | 0.19 | 0.00 | 0.00 | 0.00 | 0.00 | 0.22 | 0.00 |
| 220 | \ | 0.19 | 0.00 | 0.00 | 0.00 | 0.00 | 0.22 | 0.00 |
| 52 | digit 4 | 0.17 | 0.18 | 0.18 | 0.18 | 0.17 | 0.18 | 0.18 |
| 57 | digit 9 | 0.17 | 0.19 | 0.19 | 0.19 | 0.17 | 0.15 | 0.19 |
| 192 | ` | 0.17 | 0.00 | 0.00 | 0.00 | 0.00 | 0.12 | 0.00 |
| 53 | digit 5 | 0.15 | 0.17 | 0.17 | 0.17 | 0.16 | 0.16 | 0.17 |
| 219 | [ | 0.12 | 0.00 | 0.00 | 0.00 | 0.00 | 0.12 | 0.00 |
| 54 | digit 6 | 0.12 | 0.13 | 0.13 | 0.13 | 0.12 | 0.13 | 0.13 |
| 55 | digit 7 | 0.10 | 0.12 | 0.12 | 0.12 | 0.11 | 0.09 | 0.12 |
| 56 | digit 8 | 0.09 | 0.09 | 0.09 | 0.09 | 0.08 | 0.07 | 0.09 |
| 96 | keycode 96 | 0.08 | 0.00 | 0.00 | 0.00 | 0.00 | 0.08 | 0.00 |
| 97 | keycode 97 | 0.08 | 0.00 | 0.00 | 0.00 | 0.00 | 0.07 | 0.00 |
| 173 | - (FF) | 0.06 | 0.00 | 0.00 | 0.00 | 0.00 | 0.08 | 0.00 |
| 59 | keycode 59 | 0.06 | 0.00 | 0.00 | 0.00 | 0.00 | 0.08 | 0.00 |
| 45 | Insert | 0.05 | 0.00 | 0.00 | 0.00 | 0.00 | 0.05 | 0.00 |
| 144 | keycode 144 | 0.05 | 0.00 | 0.00 | 0.00 | 0.00 | 0.07 | 0.00 |
| 98 | keycode 98 | 0.05 | 0.00 | 0.00 | 0.00 | 0.00 | 0.04 | 0.00 |
| 110 | keycode 110 | 0.04 | 0.00 | 0.00 | 0.00 | 0.00 | 0.06 | 0.00 |
| 61 | keycode 61 | 0.04 | 0.00 | 0.00 | 0.00 | 0.00 | 0.05 | 0.00 |
| 221 | ] | 0.03 | 0.00 | 0.00 | 0.00 | 0.00 | 0.04 | 0.00 |
| 99 | keycode 99 | 0.03 | 0.00 | 0.00 | 0.00 | 0.00 | 0.02 | 0.00 |
| 100 | keycode 100 | 0.03 | 0.00 | 0.00 | 0.00 | 0.00 | 0.03 | 0.00 |
| 36 | Home | 0.03 | 0.00 | 0.00 | 0.00 | 0.00 | 0.02 | 0.00 |
| 105 | keycode 105 | 0.03 | 0.00 | 0.00 | 0.00 | 0.00 | 0.03 | 0.00 |
| 46 | Delete | 0.02 | 0.00 | 0.00 | 0.00 | 0.00 | 0.03 | 0.00 |

**Keycodes ≥ 10× more frequent in one group** (frequency = share of all keystrokes; present in ≥ 0.1% of the sessions of the group where it is more frequent; ∞ = absent from the other group):

| attacker | more_in | name | freq_humans | freq_bots | ratio | sessions_humans | sessions_bots |
|---|---|---|---|---|---|---|---|
| A1 | humans | Backspace | 5.280% | 0.000% | ∞ | 62.7% | 0.0% |
| A1 | humans | CapsLock | 0.508% | 0.000% | ∞ | 8.1% | 0.0% |
| A1 | humans | Left | 0.065% | 0.000% | ∞ | 0.7% | 0.0% |
| A1 | humans | ; | 0.049% | 0.000% | ∞ | 2.0% | 0.0% |
| A1 | humans | Right | 0.039% | 0.000% | ∞ | 0.5% | 0.0% |
| A1 | humans | = | 0.037% | 0.000% | ∞ | 1.4% | 0.0% |
| A1 | humans | Ctrl | 0.019% | 0.000% | ∞ | 0.5% | 0.0% |
| A1 | humans | \ | 0.019% | 0.000% | ∞ | 0.7% | 0.0% |
| A1 | humans | ` | 0.017% | 0.000% | ∞ | 0.7% | 0.0% |
| A1 | humans | [ | 0.012% | 0.000% | ∞ | 0.5% | 0.0% |
| A1 | humans | keycode 96 | 0.008% | 0.000% | ∞ | 0.3% | 0.0% |
| A1 | humans | keycode 97 | 0.008% | 0.000% | ∞ | 0.3% | 0.0% |
| A1 | humans | - (FF) | 0.006% | 0.000% | ∞ | 0.2% | 0.0% |
| A1 | humans | keycode 59 | 0.006% | 0.000% | ∞ | 0.2% | 0.0% |
| A1 | humans | Insert | 0.005% | 0.000% | ∞ | 0.2% | 0.0% |
| A1 | humans | keycode 144 | 0.005% | 0.000% | ∞ | 0.2% | 0.0% |
| A1 | humans | keycode 98 | 0.005% | 0.000% | ∞ | 0.2% | 0.0% |
| A1 | humans | keycode 110 | 0.004% | 0.000% | ∞ | 0.2% | 0.0% |
| A1 | humans | keycode 61 | 0.004% | 0.000% | ∞ | 0.1% | 0.0% |
| A1 | humans | ] | 0.003% | 0.000% | ∞ | 0.2% | 0.0% |
| A1 | humans | keycode 99 | 0.003% | 0.000% | ∞ | 0.2% | 0.0% |
| A1 | humans | keycode 100 | 0.003% | 0.000% | ∞ | 0.1% | 0.0% |
| A1 | humans | Home | 0.003% | 0.000% | ∞ | 0.1% | 0.0% |
| A1 | humans | keycode 105 | 0.003% | 0.000% | ∞ | 0.1% | 0.0% |
| A1 | humans | Delete | 0.002% | 0.000% | ∞ | 0.1% | 0.0% |
| A2 | humans | Backspace | 5.280% | 0.000% | ∞ | 62.7% | 0.0% |
| A2 | humans | CapsLock | 0.508% | 0.000% | ∞ | 8.1% | 0.0% |
| A2 | humans | Left | 0.065% | 0.000% | ∞ | 0.7% | 0.0% |
| A2 | humans | ; | 0.049% | 0.000% | ∞ | 2.0% | 0.0% |
| A2 | humans | Right | 0.039% | 0.000% | ∞ | 0.5% | 0.0% |
| A2 | humans | = | 0.037% | 0.000% | ∞ | 1.4% | 0.0% |
| A2 | humans | Ctrl | 0.019% | 0.000% | ∞ | 0.5% | 0.0% |
| A2 | humans | \ | 0.019% | 0.000% | ∞ | 0.7% | 0.0% |
| A2 | humans | ` | 0.017% | 0.000% | ∞ | 0.7% | 0.0% |
| A2 | humans | [ | 0.012% | 0.000% | ∞ | 0.5% | 0.0% |
| A2 | humans | keycode 96 | 0.008% | 0.000% | ∞ | 0.3% | 0.0% |
| A2 | humans | keycode 97 | 0.008% | 0.000% | ∞ | 0.3% | 0.0% |
| A2 | humans | - (FF) | 0.006% | 0.000% | ∞ | 0.2% | 0.0% |
| A2 | humans | keycode 59 | 0.006% | 0.000% | ∞ | 0.2% | 0.0% |
| A2 | humans | Insert | 0.005% | 0.000% | ∞ | 0.2% | 0.0% |
| A2 | humans | keycode 144 | 0.005% | 0.000% | ∞ | 0.2% | 0.0% |
| A2 | humans | keycode 98 | 0.005% | 0.000% | ∞ | 0.2% | 0.0% |
| A2 | humans | keycode 110 | 0.004% | 0.000% | ∞ | 0.2% | 0.0% |
| A2 | humans | keycode 61 | 0.004% | 0.000% | ∞ | 0.1% | 0.0% |
| A2 | humans | ] | 0.003% | 0.000% | ∞ | 0.2% | 0.0% |
| A2 | humans | keycode 99 | 0.003% | 0.000% | ∞ | 0.2% | 0.0% |
| A2 | humans | keycode 100 | 0.003% | 0.000% | ∞ | 0.1% | 0.0% |
| A2 | humans | Home | 0.003% | 0.000% | ∞ | 0.1% | 0.0% |
| A2 | humans | keycode 105 | 0.003% | 0.000% | ∞ | 0.1% | 0.0% |
| A2 | humans | Delete | 0.002% | 0.000% | ∞ | 0.1% | 0.0% |
| A3 | humans | Backspace | 5.280% | 0.000% | ∞ | 62.7% | 0.0% |
| A3 | humans | CapsLock | 0.508% | 0.000% | ∞ | 8.1% | 0.0% |
| A3 | humans | Left | 0.065% | 0.000% | ∞ | 0.7% | 0.0% |
| A3 | humans | ; | 0.049% | 0.000% | ∞ | 2.0% | 0.0% |
| A3 | humans | Right | 0.039% | 0.000% | ∞ | 0.5% | 0.0% |
| A3 | humans | = | 0.037% | 0.000% | ∞ | 1.4% | 0.0% |
| A3 | humans | Ctrl | 0.019% | 0.000% | ∞ | 0.5% | 0.0% |
| A3 | humans | \ | 0.019% | 0.000% | ∞ | 0.7% | 0.0% |
| A3 | humans | ` | 0.017% | 0.000% | ∞ | 0.7% | 0.0% |
| A3 | humans | [ | 0.012% | 0.000% | ∞ | 0.5% | 0.0% |
| A3 | humans | keycode 96 | 0.008% | 0.000% | ∞ | 0.3% | 0.0% |
| A3 | humans | keycode 97 | 0.008% | 0.000% | ∞ | 0.3% | 0.0% |
| A3 | humans | - (FF) | 0.006% | 0.000% | ∞ | 0.2% | 0.0% |
| A3 | humans | keycode 59 | 0.006% | 0.000% | ∞ | 0.2% | 0.0% |
| A3 | humans | Insert | 0.005% | 0.000% | ∞ | 0.2% | 0.0% |
| A3 | humans | keycode 144 | 0.005% | 0.000% | ∞ | 0.2% | 0.0% |
| A3 | humans | keycode 98 | 0.005% | 0.000% | ∞ | 0.2% | 0.0% |
| A3 | humans | keycode 110 | 0.004% | 0.000% | ∞ | 0.2% | 0.0% |
| A3 | humans | keycode 61 | 0.004% | 0.000% | ∞ | 0.1% | 0.0% |
| A3 | humans | ] | 0.003% | 0.000% | ∞ | 0.2% | 0.0% |
| A3 | humans | keycode 99 | 0.003% | 0.000% | ∞ | 0.2% | 0.0% |
| A3 | humans | keycode 100 | 0.003% | 0.000% | ∞ | 0.1% | 0.0% |
| A3 | humans | Home | 0.003% | 0.000% | ∞ | 0.1% | 0.0% |
| A3 | humans | keycode 105 | 0.003% | 0.000% | ∞ | 0.1% | 0.0% |
| A3 | humans | Delete | 0.002% | 0.000% | ∞ | 0.1% | 0.0% |
| A4 | humans | CapsLock | 0.508% | 0.000% | ∞ | 8.1% | 0.0% |
| A4 | humans | Left | 0.065% | 0.000% | ∞ | 0.7% | 0.0% |
| A4 | humans | ; | 0.049% | 0.000% | ∞ | 2.0% | 0.0% |
| A4 | humans | Right | 0.039% | 0.000% | ∞ | 0.5% | 0.0% |
| A4 | humans | = | 0.037% | 0.000% | ∞ | 1.4% | 0.0% |
| A4 | humans | Ctrl | 0.019% | 0.000% | ∞ | 0.5% | 0.0% |
| A4 | humans | \ | 0.019% | 0.000% | ∞ | 0.7% | 0.0% |
| A4 | humans | ` | 0.017% | 0.000% | ∞ | 0.7% | 0.0% |
| A4 | humans | [ | 0.012% | 0.000% | ∞ | 0.5% | 0.0% |
| A4 | humans | keycode 96 | 0.008% | 0.000% | ∞ | 0.3% | 0.0% |
| A4 | humans | keycode 97 | 0.008% | 0.000% | ∞ | 0.3% | 0.0% |
| A4 | humans | - (FF) | 0.006% | 0.000% | ∞ | 0.2% | 0.0% |
| A4 | humans | keycode 59 | 0.006% | 0.000% | ∞ | 0.2% | 0.0% |
| A4 | humans | Insert | 0.005% | 0.000% | ∞ | 0.2% | 0.0% |
| A4 | humans | keycode 144 | 0.005% | 0.000% | ∞ | 0.2% | 0.0% |
| A4 | humans | keycode 98 | 0.005% | 0.000% | ∞ | 0.2% | 0.0% |
| A4 | humans | keycode 110 | 0.004% | 0.000% | ∞ | 0.2% | 0.0% |
| A4 | humans | keycode 61 | 0.004% | 0.000% | ∞ | 0.1% | 0.0% |
| A4 | humans | ] | 0.003% | 0.000% | ∞ | 0.2% | 0.0% |
| A4 | humans | keycode 99 | 0.003% | 0.000% | ∞ | 0.2% | 0.0% |
| A4 | humans | keycode 100 | 0.003% | 0.000% | ∞ | 0.1% | 0.0% |
| A4 | humans | Home | 0.003% | 0.000% | ∞ | 0.1% | 0.0% |
| A4 | humans | keycode 105 | 0.003% | 0.000% | ∞ | 0.1% | 0.0% |
| A4 | humans | Delete | 0.002% | 0.000% | ∞ | 0.1% | 0.0% |
| A5b | humans | Backspace | 5.280% | 0.000% | ∞ | 62.7% | 0.0% |
| A5b | humans | CapsLock | 0.508% | 0.000% | ∞ | 8.1% | 0.0% |
| A5b | humans | Left | 0.065% | 0.000% | ∞ | 0.7% | 0.0% |
| A5b | humans | ; | 0.049% | 0.000% | ∞ | 2.0% | 0.0% |
| A5b | humans | Right | 0.039% | 0.000% | ∞ | 0.5% | 0.0% |
| A5b | humans | = | 0.037% | 0.000% | ∞ | 1.4% | 0.0% |
| A5b | humans | Ctrl | 0.019% | 0.000% | ∞ | 0.5% | 0.0% |
| A5b | humans | \ | 0.019% | 0.000% | ∞ | 0.7% | 0.0% |
| A5b | humans | ` | 0.017% | 0.000% | ∞ | 0.7% | 0.0% |
| A5b | humans | [ | 0.012% | 0.000% | ∞ | 0.5% | 0.0% |
| A5b | humans | keycode 96 | 0.008% | 0.000% | ∞ | 0.3% | 0.0% |
| A5b | humans | keycode 97 | 0.008% | 0.000% | ∞ | 0.3% | 0.0% |
| A5b | humans | - (FF) | 0.006% | 0.000% | ∞ | 0.2% | 0.0% |
| A5b | humans | keycode 59 | 0.006% | 0.000% | ∞ | 0.2% | 0.0% |
| A5b | humans | Insert | 0.005% | 0.000% | ∞ | 0.2% | 0.0% |
| A5b | humans | keycode 144 | 0.005% | 0.000% | ∞ | 0.2% | 0.0% |
| A5b | humans | keycode 98 | 0.005% | 0.000% | ∞ | 0.2% | 0.0% |
| A5b | humans | keycode 110 | 0.004% | 0.000% | ∞ | 0.2% | 0.0% |
| A5b | humans | keycode 61 | 0.004% | 0.000% | ∞ | 0.1% | 0.0% |
| A5b | humans | ] | 0.003% | 0.000% | ∞ | 0.2% | 0.0% |
| A5b | humans | keycode 99 | 0.003% | 0.000% | ∞ | 0.2% | 0.0% |
| A5b | humans | keycode 100 | 0.003% | 0.000% | ∞ | 0.1% | 0.0% |
| A5b | humans | Home | 0.003% | 0.000% | ∞ | 0.1% | 0.0% |
| A5b | humans | keycode 105 | 0.003% | 0.000% | ∞ | 0.1% | 0.0% |
| A5b | humans | Delete | 0.002% | 0.000% | ∞ | 0.1% | 0.0% |

**Keycode bigrams ≥ 10× more frequent in one group** (same rule): 1041 bigrams in total (`realism_enriched_bigrams.csv` lists all). Counts per attacker and direction:

| attacker | more_in | n |
|---|---|---|
| A1 | humans | 210 |
| A2 | humans | 210 |
| A3 | humans | 210 |
| A4 | bots | 77 |
| A4 | humans | 125 |
| A5b | humans | 209 |

The 15 most frequent (in the group where they are enriched) per attacker and direction:

| attacker | more_in | name | freq_humans | freq_bots | ratio | sessions_humans | sessions_bots |
|---|---|---|---|---|---|---|---|
| A1 | humans | Backspace -> Backspace | 2.480% | 0.000% | ∞ | 39.2% | 0.0% |
| A1 | humans | Space -> Backspace | 0.423% | 0.000% | ∞ | 16.2% | 0.0% |
| A1 | humans | E -> Backspace | 0.253% | 0.000% | ∞ | 10.6% | 0.0% |
| A1 | humans | Backspace -> E | 0.218% | 0.000% | ∞ | 9.2% | 0.0% |
| A1 | humans | Backspace -> Shift | 0.212% | 0.000% | ∞ | 8.8% | 0.0% |
| A1 | humans | Backspace -> T | 0.183% | 0.000% | ∞ | 7.9% | 0.0% |
| A1 | humans | T -> Backspace | 0.181% | 0.000% | ∞ | 7.8% | 0.0% |
| A1 | humans | Backspace -> O | 0.173% | 0.000% | ∞ | 7.6% | 0.0% |
| A1 | humans | A -> Backspace | 0.164% | 0.000% | ∞ | 7.1% | 0.0% |
| A1 | humans | I -> Backspace | 0.164% | 0.000% | ∞ | 7.1% | 0.0% |
| A1 | humans | S -> Backspace | 0.162% | 0.000% | ∞ | 7.0% | 0.0% |
| A1 | humans | Backspace -> I | 0.161% | 0.000% | ∞ | 7.1% | 0.0% |
| A1 | humans | O -> Backspace | 0.150% | 0.000% | ∞ | 6.6% | 0.0% |
| A1 | humans | Backspace -> A | 0.150% | 0.000% | ∞ | 6.4% | 0.0% |
| A1 | humans | Backspace -> Space | 0.144% | 0.000% | ∞ | 6.5% | 0.0% |
| A2 | humans | Backspace -> Backspace | 2.480% | 0.000% | ∞ | 39.2% | 0.0% |
| A2 | humans | Space -> Backspace | 0.423% | 0.000% | ∞ | 16.2% | 0.0% |
| A2 | humans | E -> Backspace | 0.253% | 0.000% | ∞ | 10.6% | 0.0% |
| A2 | humans | Backspace -> E | 0.218% | 0.000% | ∞ | 9.2% | 0.0% |
| A2 | humans | Backspace -> Shift | 0.212% | 0.000% | ∞ | 8.8% | 0.0% |
| A2 | humans | Backspace -> T | 0.183% | 0.000% | ∞ | 7.9% | 0.0% |
| A2 | humans | T -> Backspace | 0.181% | 0.000% | ∞ | 7.8% | 0.0% |
| A2 | humans | Backspace -> O | 0.173% | 0.000% | ∞ | 7.6% | 0.0% |
| A2 | humans | A -> Backspace | 0.164% | 0.000% | ∞ | 7.1% | 0.0% |
| A2 | humans | I -> Backspace | 0.164% | 0.000% | ∞ | 7.1% | 0.0% |
| A2 | humans | S -> Backspace | 0.162% | 0.000% | ∞ | 7.0% | 0.0% |
| A2 | humans | Backspace -> I | 0.161% | 0.000% | ∞ | 7.1% | 0.0% |
| A2 | humans | O -> Backspace | 0.150% | 0.000% | ∞ | 6.6% | 0.0% |
| A2 | humans | Backspace -> A | 0.150% | 0.000% | ∞ | 6.4% | 0.0% |
| A2 | humans | Backspace -> Space | 0.144% | 0.000% | ∞ | 6.5% | 0.0% |
| A3 | humans | Backspace -> Backspace | 2.480% | 0.000% | ∞ | 39.2% | 0.0% |
| A3 | humans | Space -> Backspace | 0.423% | 0.000% | ∞ | 16.2% | 0.0% |
| A3 | humans | E -> Backspace | 0.253% | 0.000% | ∞ | 10.6% | 0.0% |
| A3 | humans | Backspace -> E | 0.218% | 0.000% | ∞ | 9.2% | 0.0% |
| A3 | humans | Backspace -> Shift | 0.212% | 0.000% | ∞ | 8.8% | 0.0% |
| A3 | humans | Backspace -> T | 0.183% | 0.000% | ∞ | 7.9% | 0.0% |
| A3 | humans | T -> Backspace | 0.181% | 0.000% | ∞ | 7.8% | 0.0% |
| A3 | humans | Backspace -> O | 0.173% | 0.000% | ∞ | 7.6% | 0.0% |
| A3 | humans | A -> Backspace | 0.164% | 0.000% | ∞ | 7.1% | 0.0% |
| A3 | humans | I -> Backspace | 0.164% | 0.000% | ∞ | 7.1% | 0.0% |
| A3 | humans | S -> Backspace | 0.162% | 0.000% | ∞ | 7.0% | 0.0% |
| A3 | humans | Backspace -> I | 0.161% | 0.000% | ∞ | 7.1% | 0.0% |
| A3 | humans | O -> Backspace | 0.150% | 0.000% | ∞ | 6.6% | 0.0% |
| A3 | humans | Backspace -> A | 0.150% | 0.000% | ∞ | 6.4% | 0.0% |
| A3 | humans | Backspace -> Space | 0.144% | 0.000% | ∞ | 6.5% | 0.0% |
| A4 | bots | Z -> Backspace | 0.003% | 0.053% | 16.0 | 0.2% | 2.4% |
| A4 | bots | Q -> Backspace | 0.004% | 0.048% | 11.7 | 0.2% | 2.2% |
| A4 | bots | Space -> Z | 0.001% | 0.026% | 20.7 | 0.1% | 1.2% |
| A4 | bots | H -> D | 0.001% | 0.024% | 20.9 | 0.1% | 1.1% |
| A4 | bots | J -> Space | 0.002% | 0.021% | 12.1 | 0.1% | 1.0% |
| A4 | bots | F -> H | 0.001% | 0.017% | 16.4 | 0.0% | 0.8% |
| A4 | bots | J -> N | 0.001% | 0.014% | 16.5 | 0.0% | 0.6% |
| A4 | bots | H -> K | 0.000% | 0.013% | 38.1 | 0.0% | 0.6% |
| A4 | bots | W -> D | 0.001% | 0.013% | 18.9 | 0.0% | 0.6% |
| A4 | bots | E -> J | 0.001% | 0.013% | 14.5 | 0.0% | 0.6% |
| A4 | bots | Z -> Space | 0.001% | 0.011% | 18.4 | 0.0% | 0.5% |
| A4 | bots | F -> K | 0.000% | 0.010% | 71.4 | 0.0% | 0.5% |
| A4 | bots | H -> Q | 0.000% | 0.010% | 68.7 | 0.0% | 0.5% |
| A4 | bots | V -> S | 0.000% | 0.009% | 21.4 | 0.0% | 0.4% |
| A4 | bots | H -> J | 0.001% | 0.009% | 12.3 | 0.0% | 0.4% |
| A4 | humans | Backspace -> Space | 0.144% | 0.000% | ∞ | 6.5% | 0.0% |
| A4 | humans | . -> Space | 0.105% | 0.002% | 43.3 | 4.9% | 0.1% |
| A4 | humans | Backspace -> . | 0.091% | 0.000% | ∞ | 4.3% | 0.0% |
| A4 | humans | Space -> CapsLock | 0.085% | 0.000% | ∞ | 2.7% | 0.0% |
| A4 | humans | CapsLock -> I | 0.061% | 0.000% | ∞ | 2.6% | 0.0% |
| A4 | humans | Shift -> Shift | 0.051% | 0.000% | ∞ | 2.2% | 0.0% |
| A4 | humans | Backspace -> ' | 0.049% | 0.000% | ∞ | 2.2% | 0.0% |
| A4 | humans | CapsLock -> H | 0.048% | 0.000% | ∞ | 2.1% | 0.0% |
| A4 | humans | Left -> Left | 0.048% | 0.000% | ∞ | 0.5% | 0.0% |
| A4 | humans | I -> CapsLock | 0.045% | 0.000% | ∞ | 1.9% | 0.0% |
| A4 | humans | , -> Backspace | 0.044% | 0.004% | 10.6 | 2.0% | 0.2% |
| A4 | humans | CapsLock -> A | 0.040% | 0.000% | ∞ | 1.6% | 0.0% |
| A4 | humans | Backspace -> , | 0.039% | 0.000% | ∞ | 1.8% | 0.0% |
| A4 | humans | CapsLock -> E | 0.039% | 0.000% | ∞ | 1.6% | 0.0% |
| A4 | humans | CapsLock -> T | 0.038% | 0.000% | ∞ | 1.7% | 0.0% |
| A5b | humans | Backspace -> Backspace | 2.480% | 0.000% | ∞ | 39.2% | 0.0% |
| A5b | humans | Space -> Backspace | 0.423% | 0.000% | ∞ | 16.2% | 0.0% |
| A5b | humans | E -> Backspace | 0.253% | 0.000% | ∞ | 10.6% | 0.0% |
| A5b | humans | Backspace -> E | 0.218% | 0.000% | ∞ | 9.2% | 0.0% |
| A5b | humans | Backspace -> Shift | 0.212% | 0.000% | ∞ | 8.8% | 0.0% |
| A5b | humans | Backspace -> T | 0.183% | 0.000% | ∞ | 7.9% | 0.0% |
| A5b | humans | T -> Backspace | 0.181% | 0.000% | ∞ | 7.8% | 0.0% |
| A5b | humans | Backspace -> O | 0.173% | 0.000% | ∞ | 7.6% | 0.0% |
| A5b | humans | A -> Backspace | 0.164% | 0.000% | ∞ | 7.1% | 0.0% |
| A5b | humans | I -> Backspace | 0.164% | 0.000% | ∞ | 7.1% | 0.0% |
| A5b | humans | S -> Backspace | 0.162% | 0.000% | ∞ | 7.0% | 0.0% |
| A5b | humans | Backspace -> I | 0.161% | 0.000% | ∞ | 7.1% | 0.0% |
| A5b | humans | O -> Backspace | 0.150% | 0.000% | ∞ | 6.6% | 0.0% |
| A5b | humans | Backspace -> A | 0.150% | 0.000% | ∞ | 6.4% | 0.0% |
| A5b | humans | Backspace -> Space | 0.144% | 0.000% | ∞ | 6.5% | 0.0% |

## 2. Swap test (TEST split, frozen weights, frozen τ10)

Each test bot is paired with the test human it was generated for (same session, so the same target sentence). S1 = human keys + human timing (all test humans), S2 = bot keys + bot timing (the original attack), S3 = human keys + bot timing, S4 = bot keys + human timing; S3/S4 are aligned by position and truncated to the shorter length. Controls (not in the spec): S1t / S2t = human / bot truncated to the same pair length.

**Length mismatch and alignment:**

| attacker | pairs | length mismatch | bot longer / shorter | |diff| mean / median / p95 / max (mismatched) | keys dropped human / bot | aligned positions with the same key | identical key sequences |
|---|---|---|---|---|---|---|---|
| A1 | 43,248 | 32,064 (74.1%) | 2,995 / 29,069 | 7.3 / 5 / 22 / 122 | 11.0% / 0.2% | 51.1% | 9,670 |
| A2 | 43,248 | 32,064 (74.1%) | 2,995 / 29,069 | 7.3 / 5 / 22 / 122 | 11.0% / 0.2% | 51.1% | 9,670 |
| A3 | 43,248 | 32,064 (74.1%) | 2,995 / 29,069 | 7.3 / 5 / 22 / 122 | 11.0% / 0.2% | 51.1% | 9,670 |
| A4 | 43,248 | 36,705 (84.9%) | 17,235 / 19,470 | 7.5 / 5 / 22 / 120 | 7.3% / 5.9% | 37.3% | 3,502 |
| A5a | 43,248 | 42,581 (98.5%) | 21,337 / 21,244 | 21.2 / 18 / 50 / 142 | 21.4% / 21.6% | 8.7% | 2 |
| A5b | 43,248 | 32,064 (74.1%) | 2,995 / 29,069 | 7.3 / 5 / 22 / 122 | 11.0% / 0.2% | 51.1% | 9,521 |

**Caveat (found in the quick run, before the full run):** positional alignment pairs each timing with the key at the same POSITION. After the first place where the human corrected a typo (or the replayed human typed differently), the positions no longer carry the same character. The 'aligned positions with the same key' column shows how often they do. S3 and S4 therefore combine the keys of one sequence with timings that belong to OTHER keys. A detector that models timing *given the key* will flag that combination whoever made the keys. A5a (a verbatim replay of another human typing the same sentence) is the built-in control: its S3 and S4 are human keys + human timing from two different people.

**D4-primary**: APCER@τ10 (attacks accepted) and AUC vs S1; mean ± sd over seeds 42/43/44:

| attacker | S1 BPCER@τ10 | S2 APCER | S3 APCER | S4 APCER | S2t APCER | S1t APCER | S2 AUC | S3 AUC | S4 AUC | S2t AUC | S1t AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 | 10.5% ± 0.1 | 64.0% ± 1.3 | 39.5% ± 1.4 | 44.2% ± 0.1 | 64.2% ± 1.3 | 88.9% ± 0.1 | 0.834 ± 0.003 | 0.890 ± 0.003 | 0.763 ± 0.001 | 0.833 ± 0.003 | 0.499 ± 0.000 |
| A2 | 10.5% ± 0.1 | 0.6% ± 0.0 | 0.3% ± 0.0 | 44.2% ± 0.1 | 0.6% ± 0.0 | 88.9% ± 0.1 | 0.991 ± 0.000 | 0.995 ± 0.000 | 0.763 ± 0.001 | 0.991 ± 0.000 | 0.499 ± 0.000 |
| A3 | 10.5% ± 0.1 | 3.0% ± 0.0 | 1.4% ± 0.0 | 44.2% ± 0.1 | 3.0% ± 0.0 | 88.9% ± 0.1 | 0.977 ± 0.000 | 0.987 ± 0.000 | 0.763 ± 0.001 | 0.977 ± 0.000 | 0.499 ± 0.000 |
| A4 | 10.5% ± 0.1 | 1.3% ± 0.0 | 0.5% ± 0.0 | 28.0% ± 0.1 | 1.5% ± 0.0 | 89.0% ± 0.1 | 0.985 ± 0.000 | 0.994 ± 0.000 | 0.861 ± 0.001 | 0.985 ± 0.000 | 0.499 ± 0.000 |
| A5a | 10.5% ± 0.1 | 89.7% ± 0.1 | 7.8% ± 0.3 | 7.8% ± 0.2 | 88.1% ± 0.0 | 87.7% ± 0.0 | 0.494 ± 0.000 | 0.967 ± 0.001 | 0.967 ± 0.001 | 0.505 ± 0.000 | 0.510 ± 0.000 |
| A5b | 10.5% ± 0.1 | 7.5% ± 0.2 | 5.2% ± 0.2 | 44.2% ± 0.1 | 7.6% ± 0.2 | 88.9% ± 0.1 | 0.968 ± 0.001 | 0.976 ± 0.001 | 0.763 ± 0.001 | 0.968 ± 0.001 | 0.499 ± 0.000 |

**D5 (E5, own attacker)**: APCER@τ10 (attacks accepted) and AUC vs S1; mean ± sd over seeds 42/43/44 (each D5 model tested only on the attacker it was trained on):

| attacker | S1 BPCER@τ10 | S2 APCER | S3 APCER | S4 APCER | S2t APCER | S1t APCER | S2 AUC | S3 AUC | S4 AUC | S2t AUC | S1t AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 | 10.7% ± 0.2 | 0.0% ± 0.0 | 27.1% ± 6.4 | 68.6% ± 1.5 | 0.0% ± 0.0 | 88.7% ± 0.2 | 1.000 ± 0.000 | 0.902 ± 0.034 | 0.713 ± 0.025 | 1.000 ± 0.000 | 0.513 ± 0.007 |
| A2 | 10.6% ± 0.0 | 0.0% ± 0.0 | 50.5% ± 2.3 | 51.1% ± 1.1 | 0.0% ± 0.0 | 88.8% ± 0.1 | 0.998 ± 0.000 | 0.741 ± 0.021 | 0.855 ± 0.006 | 0.998 ± 0.000 | 0.506 ± 0.005 |
| A3 | 10.3% ± 0.1 | 0.1% ± 0.0 | 57.0% ± 2.5 | 50.1% ± 2.9 | 0.2% ± 0.0 | 89.3% ± 0.1 | 0.997 ± 0.001 | 0.719 ± 0.034 | 0.865 ± 0.009 | 0.997 ± 0.001 | 0.496 ± 0.001 |
| A4 | 10.6% ± 0.1 | 0.4% ± 0.0 | 7.7% ± 1.1 | 54.0% ± 0.4 | 2.5% ± 0.1 | 91.3% ± 0.1 | 0.996 ± 0.000 | 0.965 ± 0.006 | 0.763 ± 0.005 | 0.988 ± 0.000 | 0.472 ± 0.003 |
| A5b | 10.4% ± 0.1 | 0.1% ± 0.0 | 55.2% ± 1.1 | 24.7% ± 0.1 | 0.4% ± 0.1 | 89.2% ± 0.1 | 0.998 ± 0.000 | 0.765 ± 0.040 | 0.935 ± 0.001 | 0.997 ± 0.000 | 0.500 ± 0.005 |

**Share of the S2 effect reproduced by each swap.** Effect(S) = AUC(S1 vs S) − 0.5 (and, second, flag rate(S) − flag rate(S1)); share = Effect(S3) / Effect(S2) (timing) and Effect(S4) / Effect(S2) (keys), per seed, then mean ± sd. Undefined when |Effect(S2)| < 0.05. The shares need not add to 1 (interaction, misalignment). Label rule, fixed before the full run: *signal* if the timing share ≥ 0.75 and the key share ≤ 0.25; *artefact* if the key share ≥ 0.75 and the timing share ≤ 0.25; *mixed* otherwise.

| detector | attacker | Effect(S2) AUC | timing share (S3) AUC | key share (S4) AUC | timing share (S3) flag | key share (S4) flag | truncation control (S2t) AUC | label (AUC shares) |
|---|---|---|---|---|---|---|---|---|
| D4-primary | A1 | 0.33 ± 0.00 | 1.17 ± 0.00 | 0.79 ± 0.01 | 1.96 ± 0.05 | 1.78 ± 0.10 | 1.00 ± 0.00 | mixed |
| D4-primary | A2 | 0.49 ± 0.00 | 1.01 ± 0.00 | 0.53 ± 0.00 | 1.00 ± 0.00 | 0.51 ± 0.00 | 1.00 ± 0.00 | mixed |
| D4-primary | A3 | 0.48 ± 0.00 | 1.02 ± 0.00 | 0.55 ± 0.00 | 1.02 ± 0.00 | 0.52 ± 0.00 | 1.00 ± 0.00 | mixed |
| D4-primary | A4 | 0.49 ± 0.00 | 1.02 ± 0.00 | 0.74 ± 0.00 | 1.01 ± 0.00 | 0.70 ± 0.00 | 1.00 ± 0.00 | mixed |
| D4-primary | A5a | -0.01 ± 0.00 | — | — | — | — | — | n/a (S2 not detected) |
| D4-primary | A5b | 0.47 ± 0.00 | 1.02 ± 0.00 | 0.56 ± 0.00 | 1.03 ± 0.00 | 0.55 ± 0.00 | 1.00 ± 0.00 | mixed |
| D5 (E5, own attacker) | A1 | 0.50 ± 0.00 | 0.80 ± 0.07 | 0.43 ± 0.05 | 0.70 ± 0.07 | 0.23 ± 0.02 | 1.00 ± 0.00 | mixed |
| D5 (E5, own attacker) | A2 | 0.50 ± 0.00 | 0.48 ± 0.04 | 0.71 ± 0.01 | 0.44 ± 0.03 | 0.43 ± 0.01 | 1.00 ± 0.00 | mixed |
| D5 (E5, own attacker) | A3 | 0.50 ± 0.00 | 0.44 ± 0.07 | 0.73 ± 0.02 | 0.37 ± 0.03 | 0.44 ± 0.03 | 1.00 ± 0.00 | mixed |
| D5 (E5, own attacker) | A4 | 0.50 ± 0.00 | 0.94 ± 0.01 | 0.53 ± 0.01 | 0.92 ± 0.01 | 0.40 ± 0.00 | 0.98 ± 0.00 | mixed |
| D5 (E5, own attacker) | A5b | 0.50 ± 0.00 | 0.53 ± 0.08 | 0.87 ± 0.00 | 0.38 ± 0.01 | 0.73 ± 0.00 | 1.00 ± 0.00 | mixed |

**Identical-key-sequence pairs (addition, not in the spec).** These are the pairs where the human typed exactly the bot's keycode sequence: same length, same key at every position (so no correction, the same Shift use). Here the keys are identical by construction, so any separation of the bot from its human can only come from timing. S1 and S2 are restricted to these pairs:

| detector | attacker | identical pairs | humans BPCER@τ10 | bots APCER@τ10 | AUC |
|---|---|---|---|---|---|
| D4-primary | A1 | 9,670 | 6.8% ± 0.1 | 59.1% ± 1.4 | 0.895 ± 0.001 |
| D4-primary | A2 | 9,670 | 6.8% ± 0.1 | 0.9% ± 0.1 | 0.993 ± 0.000 |
| D4-primary | A3 | 9,670 | 6.8% ± 0.1 | 4.3% ± 0.1 | 0.983 ± 0.000 |
| D4-primary | A4 | 3,502 | 7.6% ± 0.2 | 2.7% ± 0.2 | 0.986 ± 0.000 |
| D4-primary | A5a | 2 | 0.0% ± 0.0 | 50.0% ± 0.0 | 0.500 ± 0.000 |
| D4-primary | A5b | 9,521 | 6.8% ± 0.1 | 9.0% ± 0.2 | 0.975 ± 0.001 |
| D5 (E5, own attacker) | A1 | 9,670 | 24.4% ± 0.7 | 0.0% ± 0.0 | 1.000 ± 0.000 |
| D5 (E5, own attacker) | A2 | 9,670 | 31.5% ± 0.3 | 0.0% ± 0.0 | 0.994 ± 0.000 |
| D5 (E5, own attacker) | A3 | 9,670 | 33.0% ± 0.8 | 0.1% ± 0.0 | 0.989 ± 0.002 |
| D5 (E5, own attacker) | A4 | 3,502 | 19.0% ± 0.4 | 0.6% ± 0.0 | 0.988 ± 0.001 |
| D5 (E5, own attacker) | A5b | 9,521 | 32.9% ± 1.4 | 0.2% ± 0.0 | 0.992 ± 0.001 |

## 3. Letters-only D4-primary

Session score = mean per-step NLL over the steps where both the current and the next key are a–z or space (all other steps masked; the recurrent state still sees the whole history). τ10 recomputed on val humans with the same masked score (10% of val humans rejected). Test humans: BPCER@τ10 10.6% ± 0.0; letter steps are 80.8% of human steps; sessions without any letter step: test humans 0, val humans 0 (excluded).

| attacker | APCER@τ10 original | APCER@τ10 letters-only | AUC original | AUC letters-only | letter-step share | sessions without letter steps |
|---|---|---|---|---|---|---|
| A1 | 64.0% ± 1.3 | 62.8% ± 1.6 | 0.834 ± 0.003 | 0.838 ± 0.004 | 88.3% | 0 |
| A2 | 0.6% ± 0.0 | 0.8% ± 0.0 | 0.991 ± 0.000 | 0.988 ± 0.000 | 88.3% | 0 |
| A3 | 3.0% ± 0.0 | 3.9% ± 0.1 | 0.977 ± 0.000 | 0.971 ± 0.000 | 88.3% | 0 |
| A4 | 1.3% ± 0.0 | 2.3% ± 0.1 | 0.985 ± 0.000 | 0.979 ± 0.000 | 81.7% | 0 |
| A5a | 89.7% ± 0.1 | 90.0% ± 0.1 | 0.494 ± 0.000 | 0.495 ± 0.000 | 80.8% | 0 |
| A5b | 7.5% ± 0.2 | 9.8% ± 0.3 | 0.968 ± 0.001 | 0.960 ± 0.001 | 88.3% | 0 |

## 4. Verdict

_Written by hand after reading the full-run tables above (post-hoc interpretation)._

**How to read the swap test here.** The per-cell rule labels every cell *mixed*. That label is not informative, because
the swap as specified has a confound: positional alignment. Only 51% of aligned positions carry the same key for
A1/A2/A3/A5b (37% for A4, 9% for A5a), because the human corrected typos and the bot did not. S3 and S4 therefore pair
timings with keys they were not typed for. The A5a row is the control: both of its halves are real humans typing the
same sentence, yet D4 flags its S3 and S4 at AUC 0.967, *more* than any bot's S4 (0.763–0.861). So a large S4 effect
measures the damage the swap does to the key–timing link, not the bots' keys. Timing shares above 1 (S3 flagged more
than S2) show the same thing. Two cleaner checks answer the question: (i) the identical-key-sequence pairs, where
the keys are equal by construction, and (ii) the letters-only score.

**D4 (one-class, humans only): SIGNAL.**
- **Same keys, different results.** A1, A2, A3 and A5b emit *byte-identical* key sequences (JSD to humans 0.0315 for
  all four), yet D4 accepts them at very different rates: 64.0%, 0.6%, 3.0% and 7.5%. The key sequence cannot be
  what separates them.
- **Identical-key pairs.** In the 9,670 pairs (3,502 for A4) where the human typed exactly the bot's keys, D4 still
  separates A2/A3/A4/A5b from their humans. AUC is 0.993, 0.983, 0.986 and 0.975, and APCER@τ10 is 0.9–9.0%, with
  these humans rejected at only 6.8–7.6%.
- **Letters-only score.** With every step that touches Shift, punctuation, Backspace or any other non-letter key
  masked, and τ10 re-set on val humans, performance barely moves. AUC changes by at most 0.008, and APCER@τ10 rises
  by at most 2.3 pp (A5b: 7.5% → 9.8%).

D4's detections of A2–A5b are therefore driven by timing *given* the keys. The residual A1 weakness (about 60%
accepted) is also a timing property, and is unchanged under both checks.

**D5 (supervised BiGRU, E5): MIXED — the bot detections are timing-driven, but D5 also uses an error-free-key-sequence
cue that comes from the attacker generators.**
- **Bot detections: timing.** In the identical-key pairs D5 still catches its own attacker almost perfectly: AUC
  0.988–1.000, APCER@τ10 0.0–0.6%. So its detections of A2–A5b do not need any key difference.
- **The cue: error-free humans look like bots.** The same humans who typed error-free are rejected at 24–33% at the
  frozen τ10 (A4 model: 19%), against 10.3–10.7% for all test humans. D4 rejects these humans at only 6.8–7.6%, so
  their *timing* is typical. The excess rejection is therefore most plausibly a *key-sequence* cue D5 learned from
  its training bots: "no correction, no CapsLock, no arrow key → bot". This is not proven here; the evidence is
  indirect.
- **Swap results are consistent with this.** D5 flags S4 (bot keys + human timing) at AUC 0.71–0.94 (A5b model 0.935,
  key share 0.87). S3 is often flagged less than S4 for A2, A3 and A5b. Part of this is the alignment confound above.
- **Consequence.** The cue does not inflate the reported bot detection; bots are caught with the keys held fixed. It
  costs false rejections of error-free humans, and it would let a bot that inserts corrections look more human to D5.

**Exact cause (artefact in the generators — reported, not fixed).**
- **A1, A2, A3, A5b** type the target sentence verbatim: 0 Backspace/Delete (humans 5.28 per 100 keys, 62.7% of
  sessions), 0 CapsLock (humans 0.51/100, 8.1% of sessions), no arrow, Ctrl, Home or Insert keys, and no stray
  punctuation typos (`;`, `=`, `\`, `` ` ``, `[`, keypad codes 96–110). They press exactly one Shift per
  Shift-requiring character (1.000 vs humans 0.954). As a result they are 5 keys shorter on average (43.2 vs 48.4).
  The key sequences come from `a1_gaussian.char_to_key` + the per-character Shift rows in `a1_gaussian` / `common.py`
  / `a5_replay` (A5b).
- **A4** (`a4_imitator`) adds substitution typos with Backspace (4.65/100, 55.6% of sessions). But it still has no
  CapsLock or arrow keys, and its typo model produces 77 keycode bigrams that are ≥ 10× more frequent than in humans,
  e.g. `Z → Backspace`, `Q → Backspace`, `H → D`, `F → K` (each in ≤ 2.4% of sessions).
- **No keycode is over-represented in any bot set** (every ≥ 10× keycode is *more frequent in humans*). No bot emits
  UNK or the unmapped code 229 on the test sentences.

**Shift timing** (a timing property, not a key artefact):
- A1 always releases Shift after the capital is released (100% vs humans 31%) and holds it longer (median 384 vs
  248 ms).
- A5b releases Shift before the next key press in 38.8% of cases (humans 5.5%), with a short hold (median 159 ms).
- A2–A4 never release Shift before the next press (humans 5.5%).

The identical-key-pair results include these Shift timings, so they are part of the timing signal D4 and D5 use.

## Files

`realism_*.csv`, `swap_length_mismatch.csv`, `swap_per_seed.csv`, `swap_shares_per_seed.csv`, `identical_keys_per_seed.csv`, `letters_only_per_seed.csv`, `reproduction_check.csv`, `phase3c_meta.json`; code `src/kl/phase3c/`, `scripts/40_run_phase3c.py`, `scripts/41_report_phase3c.py`; spec `docs/PHASE3C_SPEC.md`.
