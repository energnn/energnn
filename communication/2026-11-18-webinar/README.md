# Webinar: hands-on overview

Material of the EnerGNN webinar of 18 November 2026:

- the [recording](https://youtu.be/g6Ub_gZYslQ?t=40) (YouTube);
- the slides, `energnn_webinar.pdf`;
- the notebook, [`docs/hands_on_overview.ipynb`](../../docs/hands_on_overview.ipynb), rendered
  [in the documentation](https://energnn.readthedocs.io/en/latest/hands_on_overview.html);
- the data that the notebook reads, `case1354pegase.mat` and `ieee14.pkl` (see [Data](#data)).

The notebook walks through the data format, the problem interface, the model library and the training of EnerGNN,
on a voltage-control use case: the 1354-bus PEGASE network, with the gradient of the objective computed through
the adjoint of the OpenLoadFlow AC power flow.

## Environment

The use case needs `pypowsybl.sensitivity.run_adjoint`, which only exists in a development build of
pypowsybl (1.17.0.dev1, a `cp311` wheel), hence a Python 3.11 environment:

```bash
# from the root of the repository
uv venv --python 3.11 .venv-webinar
uv pip install --python .venv-webinar/bin/python -e ".[viz]" ipykernel nbconvert pandapower \
    /path/to/pypowsybl-1.17.0.dev1-cp311-cp311-linux_x86_64.whl
uv pip install --python .venv-webinar/bin/python --no-deps \
    "pypowsybl-to-energnn @ git+https://github.com/energnn/pypowsybl-to-energnn@element-classes" \
    "pandapower-to-energnn @ git+https://github.com/energnn/pandapower-to-energnn@feature/ac-load-flow-converter"
```

(`--no-deps` because these branches pin `energnn` themselves.) Then run the notebook with that
kernel, or execute it from the command line:

```bash
cd docs && JAX_PLATFORMS=cpu ../.venv-webinar/bin/python -m nbconvert --to notebook --execute --inplace hands_on_overview.ipynb
```

The whole notebook runs in about five minutes on CPU.

The documentation shows the outputs stored in the notebook, without running it.

## Data

`case1354pegase.mat` is the MATPOWER case `case1354pegase` (Josz, Fliscounakis, Maeght and Panciatici,
*AC Power Flow Data in MATPOWER and QCQP Format: iTesla, RTE Snapshots, and PEGASE*,
[arXiv:1603.01533](https://arxiv.org/abs/1603.01533)), converted from the `.m` file of the
[MATPOWER repository](https://github.com/MATPOWER/matpower/blob/master/data/case1354pegase.m) to the
`.mat` format read by pypowsybl (the numeric matrices `bus`, `gen`, `branch` and `gencost` saved with
`scipy.io.savemat` under a `mpc` struct). The data are fictitious and must not be used for operation or
planning; please cite the paper above when publishing results based on them.

`ieee14.pkl` is the ready-made graph loaded at the start of the notebook: the IEEE 14-bus network of
pypowsybl (`pypowsybl.network.create_ieee14()`), after an AC power flow, converted with
`pypowsybl-to-energnn` and saved with `Graph.to_pickle`.
