=============================================
Webinar: hands-on overview (18 November 2026)
=============================================

A hands-on overview of EnerGNN: the data format, the problem interface, the model library and the training, on a
voltage-control use case (the 1354-bus PEGASE network, with the gradient of the objective computed through the
adjoint of the OpenLoadFlow AC power flow).

Recording
=========

.. raw:: html

    <iframe src="https://www.youtube-nocookie.com/embed/g6Ub_gZYslQ?start=40" title="Recording of the EnerGNN webinar"
            style="width: 100%; aspect-ratio: 16 / 9; border: 0;" referrerpolicy="strict-origin-when-cross-origin"
            allowfullscreen loading="lazy"></iframe>

`Watch on YouTube <https://youtu.be/g6Ub_gZYslQ?t=40>`_.

Slides
======

.. raw:: html

    <iframe src="energnn_webinar.pdf" title="Slides of the EnerGNN webinar"
            style="width: 100%; aspect-ratio: 16 / 10; border: 0;" loading="lazy"></iframe>

`Download the slides <energnn_webinar.pdf>`_ (PDF).

Notebook
========

Two notebooks were showcased during the webinar:

- a notebook on the contingency screening use case, on full-scale RTE data: it is not open source, as it contains
  proprietary data;
- a hands-on overview notebook, available below. It is shown with the outputs it was saved with: the documentation
  does not run it.

.. toctree::
    :maxdepth: 1

    hands_on_overview

Running the notebook
====================

The use case needs ``pypowsybl.sensitivity.run_adjoint``, which only exists in a development build of pypowsybl
(1.17.0.dev1, a ``cp311`` wheel), hence a Python 3.11 environment:

.. code-block:: bash

    # from the root of the repository
    uv venv --python 3.11 .venv-webinar
    uv pip install --python .venv-webinar/bin/python -e ".[viz]" ipykernel nbconvert pandapower \
        /path/to/pypowsybl-1.17.0.dev1-cp311-cp311-linux_x86_64.whl
    uv pip install --python .venv-webinar/bin/python --no-deps \
        "pypowsybl-to-energnn @ git+https://github.com/energnn/pypowsybl-to-energnn@element-classes" \
        "pandapower-to-energnn @ git+https://github.com/energnn/pandapower-to-energnn@feature/ac-load-flow-converter"

(``--no-deps`` because these branches pin ``energnn`` themselves.) Then run the notebook with that kernel, or
execute it from the command line:

.. code-block:: bash

    cd docs/communication/2026-11-18-webinar
    JAX_PLATFORMS=cpu ../../../.venv-webinar/bin/python -m nbconvert --to notebook --execute --inplace hands_on_overview.ipynb

The whole notebook runs in about five minutes on CPU.

Data
====

The notebook reads two files, stored next to it.

``case1354pegase.mat`` is the MATPOWER case ``case1354pegase`` (Josz, Fliscounakis, Maeght and Panciatici,
*AC Power Flow Data in MATPOWER and QCQP Format: iTesla, RTE Snapshots, and PEGASE*,
`arXiv:1603.01533 <https://arxiv.org/abs/1603.01533>`_), converted from the ``.m`` file of the
`MATPOWER repository <https://github.com/MATPOWER/matpower/blob/master/data/case1354pegase.m>`_ to the ``.mat``
format read by pypowsybl (the numeric matrices ``bus``, ``gen``, ``branch`` and ``gencost`` saved with
``scipy.io.savemat`` under a ``mpc`` struct). The data are fictitious and must not be used for operation or planning;
please cite the paper above when publishing results based on them.

``ieee14.pkl`` is the ready-made graph loaded at the start of the notebook: the IEEE 14-bus network of pypowsybl
(``pypowsybl.network.create_ieee14()``), after an AC power flow, converted with ``pypowsybl-to-energnn`` and saved
with ``Graph.to_pickle``.
