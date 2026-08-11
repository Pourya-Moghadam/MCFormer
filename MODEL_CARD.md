# MC-Former model card

## Summary

MC-Former is a training framework for RGB video action recognition. A training-only Motion
Coupling Induction Module supervises video features using hand/object trajectory coupling. The
exported classifier accepts RGB clips only and contains neither MCIM nor pose, detection, or
tracking components.

## Intended use

- Research on action recognition and training with privileged interaction information.
- Reproduction and extension of the experiments described in the accompanying manuscript.
- RGB-only inference after a checkpoint has been explicitly exported and verified.

It is not intended for safety-critical decisions, biometric identification, surveillance
deployment, or unsupervised decisions about people.

## Inputs and outputs

The public model contract accepts normalized tensors in `B,T,C,H,W` layout, normally 32 RGB frames
at 224x224. It returns multiclass action logits and classifier-input representations. Training
additionally consumes cached targets derived from pose and object observations.

## Training data and weights

The paper studies NTU RGB+D 60, NTU RGB+D 120, and Toyota Smarthome. Their media and restricted
annotations are not included. Version 0.1.0 does not bundle trained or third-party initialization
weights. Every externally supplied checkpoint is local and content-addressed by SHA-256.

## Limitations

- Performance depends on dataset protocol fidelity and the quality of training-time pose,
  detection, and tracking observations.
- Motion coupling is a limited directional co-motion cue, not a complete representation of human
  intent, contact, object state, or causal interaction.
- Dataset biases, camera conditions, class imbalance, and annotation errors may transfer to the
  classifier.
- Attention visualizations are diagnostic associations and must not be interpreted as causal
  explanations.
- The implementation does not make manuscript result values available without running the
  documented experiments on appropriately licensed data.

## Reproducibility and evaluation

Seeds, configurations, preprocessing contracts, checkpoint identity, and artifact formats are
defined in `REPRODUCIBILITY_SPEC.md`. Evaluation reports top-1 accuracy, top-5 accuracy, mean class
accuracy, per-class accuracy, and confusion matrices as applicable. Result discrepancies should be
reported rather than adjusted to match the manuscript.

## License

Repository code is MIT-licensed. Dataset, annotation, and third-party checkpoint licenses remain
independent and must be reviewed by users.
