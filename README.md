# MMMData

Multi-Modal Memory Dataset

## Overview

This repository hosts project code and a wiki-like documentation site built with Jekyll (Just the Docs theme).

- Documentation site: https://hulacon.github.io/mmmdata/
- Docs source: `docs/`
- Project code: root-level folders and files in this repo

## Working on the docs locally

You'll need Ruby and Bundler. On macOS you can use the system Ruby or a version manager.

```sh
cd docs
bundle install
bundle exec jekyll serve
```

Then open the printed local URL (typically http://localhost:4000/mmmdata).

## Deployment

Docs are built and deployed to GitHub Pages via the workflow in `.github/workflows/pages.yml` whenever changes under `docs/` are pushed to `main`.

## Contributing

- For documentation: add pages under `docs/doc/` and follow the existing front matter examples (`title`, `nav_order`).
- For code: open a pull request with clear description and tests where applicable.

## License

See `docs/LICENSE` for the documentation template license. Project code license can be added in the root if needed.
