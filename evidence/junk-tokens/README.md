# Junk-token investigation outputs (2026-10-24)

Raw completions from `scripts/junk_test.py` (three coding tasks, fixed seeds), copied from the dev server.
File name: `junk<run>-T<temperature>-<thinking on|off>-<task>.txt`.

| Run | Engine configuration | Result |
|---|---|---|
| (no letter) | v4, auto B12X MoE kernel, DSpark; T=0 and T=1, thinking on and off | junk in code at T=0 (e.g. `inserts: int = 0都不要`, `0.02ASA`); worse at T=1 |
| A | v4, auto kernel, expert-major prefill | junk (`都不要`) |
| C | v4, auto kernel, token-chunk prefill | junk (`evictions += 1inks`) |
| D | v2 eager, deterministic M16 kernel (the bit-exact reference path), task 0 only | clean |
| E | v4, deterministic M16 kernel for decode batches <= 8 (39 tok/s) | clean, 8 code blocks |
| F | split pack + deterministic kernel, before the masked-route fix | clean code; long-prompt recall was broken |
| G | split pack, T=0.6, thinking on | no junk (no code block finished within the token limit) |
| H | split pack + deterministic kernel + masked-route fix (deployed) | clean, 8 code blocks |

Exact DSpark on/off settings of runs A and C are not recorded in these files.
