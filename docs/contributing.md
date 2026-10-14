# Contributing

Issues and pull requests are welcome at
<https://github.com/GeoPilot-AI/pumptest>.

## Set up

```
git clone https://github.com/GeoPilot-AI/pumptest
cd pumptest
pip install -e ".[dev,docs]"
python scripts/checks.py
```

`scripts/checks.py` runs every gate CI runs: lint, format, house style, types,
tests, per-module coverage, dependency licences, the documentation build and
the decision records. A pull request is mergeable when it is green.

## Adding a solution

1. Write the Laplace-domain kernel (or the closed form) in its own module.
2. Test it against a **published table** of the solution, in
   `tests/test_type_curves.py`, citing the table.
3. Add a `Model` subclass in `pumptest.models` with its parameters and bounds.
4. Add a fit test on synthetic data that recovers the parameters.
5. If a published field test analysed with that solution exists, bundle it in
   `pumptest.datasets` with its provenance and add it to the validation page.
6. Record the decision in `pm/decisions/` if it shapes the design.

## Regenerating the validation page and figures

```
python scripts/validation_report.py --write
python scripts/figures.py
```

`tests/test_docs.py` fails if `docs/validation.md` and a fresh run disagree,
and runs every code block in the README and the documentation.

## House rules

- No em-dashes or en-dashes anywhere (the `style` gate).
- Full type annotations (the `types` gate).
- Never describe unbuilt behaviour as if it exists.
- Conventional commits, one logical change each.
