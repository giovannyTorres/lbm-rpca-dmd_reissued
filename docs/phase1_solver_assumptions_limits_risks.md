# FASE 1B Solver Notes

## Assumptions

- The solver remains a minimal 2D D2Q9 BGK implementation for flow around one circular obstacle.
- The obstacle uses bounce-back no-slip treatment.
- The top and bottom boundaries are periodic.
- The inlet imposes fixed velocity `(u_in, 0)` with `rho = 1`.
- The outlet uses a simple zero-gradient copy from the previous column.
- Snapshots are stored as CSV matrices and validated for shape and numeric integrity.

## Limits

- This is not a high-fidelity reference implementation.
- Outlet and boundary handling are intentionally simple and may reflect waves or distort wakes for aggressive settings.
- The JSON parser is intentionally flat and only supports the current key/value configuration style.
- Runtime checks are basic safety guards, not a substitute for physical validation or benchmark comparison.
- CSV output is convenient for inspection but inefficient for large studies.

## Risks

- Runs close to `tau = 0.5` can destabilize quickly.
- High `u_in` increases compressibility error and can drive non-physical velocities.
- High Reynolds number on coarse grids can produce unstable or misleading wake structures.
- Small domains around the cylinder can contaminate the wake through boundary interaction.
- The current periodic `y` direction is a modeling choice and not universally appropriate.

## Safe Starter Parameters

Use these as conservative first runs before increasing aggressiveness:

- `nx = 160` to `220`
- `ny = 60` to `80`
- `obstacle_r = 6` to `10`
- `u_in = 0.03` to `0.05`
- `reynolds = 60` to `150`
- Keep `tau >= 0.53`
- Start with `iterations = 100` to `500` for smoke checks
- Use `save_stride = 20` to `100` depending on run length

If a run is unstable, reduce `u_in`, reduce `reynolds`, increase the domain, or increase `obstacle_r` while preserving a comfortable clearance to the boundaries.
