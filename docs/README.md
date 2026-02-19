# Docs Guide

This folder contains both the Sphinx **source** files and the generated GitHub Pages site.

## Layout

- `docs/source/`: editable documentation source (`.rst`, `conf.py`, CSS, static assets).
- `docs/index.html` and related `docs/_static`, `docs/_sources`, etc.: generated HTML published by GitHub Pages.
- `docs/.doctrees/`: Sphinx build cache (intermediate files).

## Key source files

- `docs/source/index.rst`: homepage content.
- `docs/source/leaderboard.rst`: leaderboard page.
- `docs/source/install/installation.rst`: setup instructions.
- `docs/source/get_started/overview.rst`: project overview.
- `docs/source/conf.py`: Sphinx configuration.
- `docs/source/_static/custom.css`: site style/theme overrides.
- `docs/source/hybridrag_logo.py`: custom `:hybridrag:` inline logo role.

## Build docs

From `docs/source/`:

```bash
make html
```

This is configured to output directly to `docs/` (not `docs/build/`), so GitHub Pages can publish from the `/docs` folder.

Alternative command:

```bash
sphinx-build -b html -d ../.doctrees . ..
```

## Clean generated site

From `docs/source/`:

```bash
make clean
```

This removes generated HTML/assets under `docs/` and the doctree cache.
