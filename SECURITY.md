# Security policy

Do not open a public issue containing credentials, private dataset paths, restricted annotations,
or unreleased checkpoints. Report a suspected vulnerability privately through GitHub's security
advisory feature for this repository.

Only the latest version on the default branch is supported. The project loads local checkpoints
only after an explicit SHA-256 check, but PyTorch checkpoint files should still be treated as
untrusted input. Use artifacts from known sources and review their licenses before use.
