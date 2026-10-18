"""Release-time de-identification of a staged copy of the dataset.

The working tree is never edited by this package. Every transform runs on a
**staged copy** of the BIDS tree and refuses to start if pointed at the
configured BIDS root (or anything inside it), at the source-data directory,
or at a directory that contains either of them.

Expected usage
--------------
1. Copy the dataset to a staging path outside the BIDS root (a cluster job;
   out of scope here). A hardlink copy (``cp -al``, ``rsync --link-dest``)
   is fine: every transform writes a new file beside the old one and renames
   it into place, so a rewritten file gets a new inode and the working-tree
   file it was linked to keeps its bytes.
2. Run the export on the staged root::

       release_export.py export <stage> \\
           --offsets-out <private-dir>/date_offsets.tsv \\
           --pronoun-report <private-dir>/pronoun_review.tsv \\
           --redactions <private-dir>/redactions.tsv \\
           --names-file <private-dir>/names.txt

   ``--dry-run`` plans every step in memory, reports what would change, and
   runs the sweep against the planned result without writing anything. A
   real run also sweeps the planned result first, and writes nothing if the
   sweep finds anything (``--commit-with-findings`` overrides that once every
   finding has been judged acceptable), so a failed run can simply be rerun.
3. Have a person read the pronoun review report; fix any grammar it flags
   with a follow-up redaction-style edit, never in the working tree.
4. Ship the stage only when the sweep exits 0.

Private outputs (the per-participant date offsets, the pronoun review
report, the flag list) contain participant-level detail and must live
outside the stage and outside the BIDS root; the tools refuse otherwise,
and never overwrite an existing private output.

Steps
-----
``dates``     shift ``sub-##/sub-##_sessions.tsv`` ``acq_time`` per
              participant so the first session falls on a fixed epoch,
              intervals preserved.
``pronouns``  neutralize gendered third-person pronouns in free-text
              columns of the debriefing table, with a review report.
``redact``    apply a human-reviewed redaction spec (``[REDACTED]`` token
              plus a ``Redaction`` note in the sidecar); ``flag`` lists
              candidate spans for the human reader (advisory only).
``sweep``     scan shipped text files for leftover identifiers.
``export``    run the four in order.
"""
