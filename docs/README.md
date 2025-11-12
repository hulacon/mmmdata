# MMMData docs

This `docs/` folder contains the source for the MMMData documentation site, built with Jekyll using the Just the Docs theme.

Live site: https://hulacon.github.io/mmmdata/

## Run locally

Prereqs: Ruby and Bundler installed.

```sh
bundle install
bundle exec jekyll serve
```

Open the URL printed in your terminal (typically http://localhost:4000/mmmdata).

## Contribute content

- Add pages under `doc/`.
- Use front matter to control navigation order:

  ```yaml
  ---
  title: My Page
  nav_order: 10
  ---
  ```

- Link pages with relative links like `[About](doc/about)`.

## Build and deploy

GitHub Actions builds and deploys the site on pushes to `main` that touch `docs/`. See `.github/workflows/pages.yml`.
