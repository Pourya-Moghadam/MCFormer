# Publishing the repository

Publish the **contents of this `code/` directory as the root of the public GitHub repository**.
Do not publish the parent manuscript workspace: it contains journal correspondence, reviewer
material, drafts, LaTeX build products, and large files unrelated to the software release.

Before placing the link in the manuscript:

1. Confirm the GitHub repository is public and its default branch contains this README at root.
2. Confirm the CI workflow passes on Python 3.11.
3. Create an immutable `v0.1.0` release from the reviewed commit.
4. Verify the release contains no dataset media, annotations, checkpoints, output directories,
   local environment, build products, secrets, or private paths.
5. Use the repository URL in the manuscript; for archival reproducibility, also cite the release
   tag or a Zenodo DOI created from that GitHub release.
6. Add the paper DOI to `CITATION.cff` after it is assigned.

The public release is intentionally code-only. Missing licensed datasets or third-party weights
are documented inputs, not files to add to Git history.
