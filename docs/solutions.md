# Solutions

| Model | Aquifer | Fitted parameters | Reference |
|---|---|---|---|
| `Theis` | Confined | T, S | Theis (1935) |
| `HantushJacob` | Leaky, aquitard without storage | T, S, B (reports c and leakance) | Hantush and Jacob (1955) |
| `PapadopulosCooper` | Confined, large-diameter well | T, S, optionally skin Sk | Papadopulos and Cooper (1967); Agarwal et al. (1970) |
| `Neuman` | Unconfined, delayed yield | T, S, Sy, Kd = Kz/Kr | Neuman (1974) |

Straight-line methods, in `pumptest.straightline`:

- `cooper_jacob`: time-drawdown line, walking the start later until u <= 0.05.
- `theis_recovery`: residual drawdown against t / t'.
- `step_drawdown`: s / Q = B + C Q, with well efficiency at each step.

All models are linear in the pumping rate, so recovery, step and other
variable-rate tests are handled by superposing the pumping `Schedule`.

## Not modelled yet

Partial penetration, aquitard storage (Hantush 1960, Neuman-Witherspoon),
Moench's gradual drainage, fractured and double-porosity media, bounded
aquifers and slug tests are not implemented. Nothing in the package does them
today.
