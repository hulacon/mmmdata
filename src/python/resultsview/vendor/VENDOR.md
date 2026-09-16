# Vendored third-party assets

The three Vega UMD builds a results page inlines (`resultsview/page.py`).
All BSD-3-Clause, University of Washington Interactive Data Lab; each
license is beside its file.

| File | Package | sha256 |
|---|---|---|
| `vega.min.js` | `vega` 6.4.0, `package/build/vega.min.js` | `8f6a3587cf8d4f42c7e08120e3eb05d067e746d554e39d2dcf52acc0bd5ba28f` |
| `vega-lite.min.js` | `vega-lite` 6.4.3, `package/build/vega-lite.min.js` | `35a9821df838825b05a6a73e9414b58747a1b18321583858ed903c66393a5c7e` |
| `vega-embed.min.js` | `vega-embed` 7.2.0, `package/build/vega-embed.min.js` | `b69eac2846a0061683b7e03501790fb0bcbdb851c797c6baf3417c9d8852819e` |

Source: `https://registry.npmjs.org/<package>/-/<package>-<version>.tgz`.

Vendored rather than fetched at page load for the same reason as NiiVue
(`neuroimaging/vendor/VENDOR.md`): pages must open over `file://` and on
compute nodes with no internet. `tests/test_resultsview.py` pins the
hashes and the versions the page reports — update all three together.
