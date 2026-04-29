# Contributing to RLDX

Thanks for taking the time to look at this. RLDX is a research codebase
on its way to a public release, so the most useful contributions today
are bug reports, environment-setup feedback, and well-scoped pull
requests against open issues.

## Quick start

```bash
git clone https://github.com/RLWRLD/RLDX.git
cd RLDX
uv sync --python 3.10
uv pip install -e ".[dev]"
uv tool install ruff
uv tool install pre-commit
pre-commit install
```

`pre-commit install` wires in the same `ruff check` + `ruff format
--check` rules CI runs, scoped to `rldx/`. From this point any commit
you make is blocked locally on the same lint that the server enforces.

Verify the install:

```bash
python -c "import rldx; print(rldx.__version__)"
```

For environment-level details (CUDA, flash-attn, simulator venvs,
common pitfalls) read [`docs/installation.md`](docs/installation.md)
first.

## Reporting bugs

Open an issue at https://github.com/RLWRLD/RLDX/issues with:

1. Branch + commit hash (`git rev-parse HEAD`).
2. Exact command line and any environment variables that matter
   (`CUDA_VISIBLE_DEVICES`, `HF_HOME`, `WANDB_PROJECT`, ...).
3. Full traceback (please copy text rather than screenshot).
4. What you expected and what happened.
5. If the bug touches a checkpoint, the HF repo id (or local
   `processor_config.json` if it is private).

If the bug is environment-specific, attach the output of:

```bash
python -c "import torch, transformers, rldx; print(torch.__version__, transformers.__version__, rldx.__version__)"
nvidia-smi
nvcc --version
```

## Proposing changes

We work on the `refactoring-for-release` branch until the public
release lands. After that, the workflow will move to feature branches
off `main`.

1. Open an issue first for anything non-trivial (new training recipe,
   API change, evaluation harness extension). Drive-by bug fixes can
   skip this.
2. Branch from the most recent `refactoring-for-release` (or `main`
   after release):
   ```bash
   git fetch origin
   git checkout -b feature/short-description origin/refactoring-for-release
   ```
3. Make focused commits. The history of this branch is **micro-commit
   style**: each commit is one self-contained change with the test
   evidence in the message. See `git log --oneline` for examples.
4. Run the local checks before pushing:
   ```bash
   uv tool run ruff check rldx/
   uv tool run ruff format --check rldx/
   ```
5. Push and open a PR against `refactoring-for-release`. Link the
   issue if there is one.

## Coding conventions

- **Lint:** `ruff` with the config in `pyproject.toml` (`I001`, `F`, `E`,
  line length 100). The CI workflow at `.github/workflows/ci.yml`
  enforces both `ruff check rldx/` and `ruff format --check rldx/`.
  Local pre-commit mirrors that.
- **Imports:** isort-style ordering is handled by ruff. Side-effect
  imports inside `__init__.py` are exempt from `F401` via
  `[tool.ruff.lint.per-file-ignores]`.
- **Type hints:** new code should be typed where it is not painful.
  No `mypy` enforcement yet.
- **Docstrings:** Google-style for new public functions. Existing
  modules have a mix; do not chase a global rewrite.
- **No print statements** in library code. Use
  `from rldx.utils.dist import rank_zero_print as _print` so
  multi-rank training does not get spammed.
- **No dead code paths.** If you remove a feature, remove its branches
  too. The release-prep refactor turned every dormant
  `NameError`-on-config-change branch into an explicit
  `NotImplementedError`; please follow the same pattern instead of
  leaving silent fall-throughs.

## Testing

When you change something user-visible, please add a test alongside
your change:

- **Processor / config** changes should land with a save/load roundtrip
  test that pins the new field shape.
- **New training CLI flags** should be referenced in
  `docs/training.md` so reviewers know how to invoke them.
- **Model / dataset** behaviour changes should have a deterministic
  unit test.

## Commit messages

The branch convention is **conventional commits with a body that lists
test evidence**. Examples from the recent history:

```
fix(action_model): eliminate F821 dormant bugs for removed config paths
chore(pyproject): add description, readme, URLs, keywords, classifiers
docs: add docs/architecture.md
test(image): comprehensive image pipeline coverage (39 tests)
```

Body convention: explain the *why*, list the files touched, paste the
relevant test output (or its summary line). Co-authoring with an LLM
is allowed and tracked via the `Co-Authored-By:` trailer.

## Documentation

`docs/` is the source of truth for hands-on guides:

- [`docs/installation.md`](docs/installation.md)
- [`docs/architecture.md`](docs/architecture.md)
- [`docs/training.md`](docs/training.md)
- [`docs/evaluation.md`](docs/evaluation.md)
- [`docs/inference_server.md`](docs/inference_server.md)
- [`docs/guide/naming.md`](docs/guide/naming.md) — canonical
  terminology (cognition tokens, MSAT, motion module, action model,
  training stages, benchmark names). Consult this before introducing
  new user-facing terms in docs, commit messages, or release notes.

The README is kept short. When you change a CLI surface, file paths,
or the install flow, the matching docs/ file is the canonical place
to update.

## License

RLDX is released under the [Apache License 2.0](LICENSE). By
submitting a contribution you agree that it can be distributed
under the same license.

## Acknowledgments

Thanks to everyone running RLDX experiments internally and surfacing
the rough edges that this contributing guide tries to flatten.
