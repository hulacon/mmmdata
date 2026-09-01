# Vendored third-party assets

## niivue.umd.js

- Package: `@niivue/niivue` 0.44.2 (MIT, see `NIIVUE_LICENSE`)
- Source: `https://registry.npmjs.org/@niivue/niivue/-/niivue-0.44.2.tgz`,
  file `package/dist/niivue.umd.js`
- sha256: `43d966b756982173fdd0cd37736c348b4db02238cdcbdd21fb16714963aad3fa`

Vendored rather than fetched at page load: viewer bundles must be
self-contained (compute nodes and `file://` pages may have no internet),
and NiiVue is not on PyPI. `tests/test_viewer.py` pins the hash — update
both together when upgrading.

## viewer_template.html

Ours, not third-party — the single-file shell `viewer.py` fills in. Kept
beside the JS it inlines so the whole viewer surface lives in one place.
