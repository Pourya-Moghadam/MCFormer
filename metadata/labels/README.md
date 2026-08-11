# Label maps

Label maps are user-supplied because dataset annotations are not redistributed. Each map is a JSON
object whose keys are contiguous zero-based integer strings and whose values are the official,
non-empty class names, for example `{"0": "official class name"}`. NTU60, NTU120, and Toyota maps
must contain exactly 60, 120, and 31 entries respectively. Their identities are preserved in run
provenance.
