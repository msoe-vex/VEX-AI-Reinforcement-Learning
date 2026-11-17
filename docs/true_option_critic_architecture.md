# Option-Critic Termination

The policy selects a discrete macro-action and has a separate sigmoid
termination head with one probability per macro-action:

\[
\pi_\Omega(\omega \mid s), \qquad
\beta_\omega(s) = \sigma(z_\omega(s)).
\]

During training, the environment exposes the active macro-action as the
`active_action` observation. Busy robots receive another policy decision every
second when the action lasts at most ten seconds, or every five seconds when it
lasts longer. The sampled termination command is ignored when the proposed
macro-action matches the active one. A different action with a termination
command interrupts the current action and incurs a raw penalty of `10.0`.

`OptionCriticPPOTorchLearner` augments PPO's loss with

\[
L_\beta =
\mathbb{E}\left[
\beta_\omega(s)\left(A(s,\omega) + \eta\right)
\right],
\]

where PPO's detached advantage estimates \(Q(s,\omega)-V(s)\), and
\(\eta=1.0\) is the effective reward-scale switching cost. Only observations
with an active macro-action contribute to this loss.

The exported model appends the per-action beta probabilities after the action
logits and optional communication distribution parameters. Inference uses
`VexModelRunner.evaluate_termination(observation, active_action)` to evaluate
the active option. `get_inference` also returns the beta value and a
`should_stop` flag; that flag is true only when beta is at least `0.5` and the
sampled replacement action differs from the active action.
When an active action continues, `get_inference` returns an empty split-action
list so the caller has no replacement payload to dispatch.

Models trained before the termination head was added must be retrained and
re-exported. The Nano evaluator and V5 preemption dispatcher are separate
runtime components and are not implemented in this repository.
