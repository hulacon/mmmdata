---
title: Getting Started
nav_order: 1
---

This page helps you get oriented with MMMData and this repository.

## Repository layout

- Project code and data-related scripts live at the repository root and subfolders.
- Documentation lives under `docs/` and is built with Jekyll using the Just the Docs theme.

## View the docs online

Once deployed, the documentation site is available at:

- https://hulacon.github.io/mmmdata/

## Run the docs site locally

You'll need Ruby and Bundler installed. On macOS:

1. Install Bundler (once):

   ```sh
   gem install bundler
   ```

2. From the repository root:

   ```sh
   cd docs
   bundle install
   bundle exec jekyll serve
   ```

3. Open http://localhost:4000/mmmdata (or the URL printed in your terminal).

## Contributing to docs

- Add or edit pages under `docs/doc/`.
- Use front matter to set ordering and hierarchy:
  
  ```yaml
  ---
  title: My Page
  nav_order: 10
  parent: Section Name # optional; if the parent page has `has_children: true`
  ---
  ```

- Link to other pages with relative paths, e.g. `[About](about)`.

See the [About](about) page to learn more about the project.
