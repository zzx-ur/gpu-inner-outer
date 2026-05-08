# Algorithm Documentation

## Overview

`cuda_symbolic_control` implements rule-based grid symbolic control using finite abstraction and bounded backward reachability for K-invariance verification.

## System Model

The 4D unicycle dynamics with disturbance:
```
x_next = f(x, u) + w
```

Nominal integration uses fixed-step Euler with specified substeps. Over-approximation is computed via linearization:
```
successor_box_radius = |Jf_x| * (cell_eta/2) + |Jf_w| * (disturbance_half_width/2)
```

where `Jf_x` is the Jacobian w.r.t. state and `Jf_w` w.r.t. disturbance.

**State constraints**: Enforced during abstraction — any (state, input) pair whose successor box contains invalid states (violating map boundaries, obstacles, or custom constraints like hyperbolic/elliptic bounds) is discarded.

## Abstraction Process

For each grid cell center `x` and input center `u`:

1. Compute axis-aligned successor box `[min, max]` covering all possible next states
2. Convert continuous box to discrete index box on the state grid
3. Validate that **all** states in the successor box satisfy system constraints
4. Store as `(min_flat, max_flat, valid)` triplet

Storage: Three arrays of size `num_states × num_inputs`:
- `pair_min_flat[p]` — flattened minimum index of successor box
- `pair_max_flat[p]` — flattened maximum index of successor box
- `pair_valid[p]` — whether this (state, input) pair produces a valid successor

## Backward Reachability Algorithm

Iterative set propagation to compute `K`-invariance:

- **Initialization**: `R0 = candidate_mask` (R the reachable set)
- **Iteration**: For each valid state `x`, check if ∃ input `u` such that
  `successor_box(x,u) ⊆ R_{k-1}` (all successor states already reachable)
- If satisfied: mark `x` as reachable and record `controller[x] = u`
- Termination: all candidate states have a controller, or iteration stalls

Per-iteration work:
1. Build 4D prefix sum of `reachable_mask` (inclusion-exclusion)
2. For each (state, input) pair: query box counts via 16-corner formula
3. Reduce to find first satisfying input per state

## Candidate and Reachable Sets

- **Candidate set**: Initial target set from which backward reachability starts
- **Reachable set**: States proven to maintain safety under some control input
- **K-invariance**: The controller ensures the system stays within safe bounds indefinitely

## State Constraints

Supported constraint types:

### Hyperbolic Constraint
```
x1² - x2² ≤ a  and  b·x2² - x1² ≤ c
```
Example (sandglass region): a=4.0, b=4.0, c=16.0

### Elliptic Constraint
```
(x1/a)² + (x2/b)² ≤ 1
```
Example: a=2.0, b=2.0 (circular region)

### No Constraint (Default)
No additional state constraints applied.

## Wrap-around Handling


Angle dimension (dim 2 by default) uses circular indexing. Prefix sums and box queries account for wrap-around by splitting跨维度的盒子为两个区间.

## Key Data Structures

| Structure | Description |
|-----------|-------------|
| `Grid4D<StateIndex>` | 4D uniform state grid with wrap support |
| `InputGrid2D` | 2D uniform input grid |
| `PairAbstraction` | `min_flat`, `max_flat`, `valid` arrays per (state, input) |
| `SolveResult` | `reachable_mask`, `controller`, `reach_step`, convergence info |

## Memory Estimation

Pre-computation estimates memory budget:

```
pair_min + pair_max + pair_valid: 9 bytes × num_pairs
pair_satisfied:                   1 byte × num_pairs
reachable_prev/next:              2 bytes × num_states
valid_mask + candidate_mask:      2 bytes × num_states
controller + reach_step:          8 bytes × num_states
prefix_valid + prefix_reachable: 16 bytes × (n0+1)(n1+1)(n2+1)(n3+1)
```

For 200 million pairs with ~10^6 states, total memory is typically under 32 GB.