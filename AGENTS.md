# Academic Development Directive

This repository contains research developments and experiments for refining and comparing elevation models.

## Mandatory Traceability

- Document each development with a clear description of the objective, assumptions, inputs, outputs, and relevant parameters.
- Record each experiment in Markdown and JSON, including runtimes, errors, warnings, dependency versions when relevant, and paths to generated artifacts.
- Keep scripts reproducible through command-line arguments and explicit default values.
- Do not process subfolders unless the experiment explicitly declares it.
- Avoid destructive changes to original data; all outputs must be written to result folders inside the workspace.

## Geospatial Data

- Always state the assumed or detected input CRS and the output CRS.
- When the CRS is not embedded in the data, document the inference used.
- Document the rasterization resolution and the per-pixel aggregation rule.
- Record comparison metrics, validity mask, number of samples, and formula used.

## Visualization

- Viewers and interactive artifacts must be regenerable from documented intermediate data.
- Simplifications or downsampling for visualization must be recorded separately from metric calculations.
