#
# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
import numpy as np


def compare_batched_graphs(*graphs, rtol=1e-6, atol=1e-6):
    """
    Compare a list of batched JaxGraph outputs.

    - Ensures same edge keys.
    - For each edge, ensures corresponding arrays have same shapes and are numerically close.
    - Also checks address_dict arrays and non_fictitious shapes.
    """

    if len(graphs) < 2:
        return

    # check keys
    keys0 = set(graphs[0].hyper_edge_sets.keys())
    for g in graphs[1:]:
        if set(g.hyper_edge_sets.keys()) != keys0:
            raise AssertionError(f"Edge keys differ: {keys0} vs {set(g.hyper_edge_sets.keys())}")

    # for each edge key, compare feature_array, address_dict, non_fictitious
    for key in keys0:
        base = graphs[0].hyper_edge_sets[key]
        # FEATURE ARRAYS (may be None)
        base_feat = base.feature_array
        base_np = None if base_feat is None else np.array(base_feat)
        for g in graphs[1:]:
            feat = g.hyper_edge_sets[key].feature_array
            feat_np = None if feat is None else np.array(feat)
            # both None -> ok
            if base_np is None and feat_np is None:
                continue
            # shape must match
            if base_np is None or feat_np is None:
                raise AssertionError(
                    f"Feature-array presence mismatch for edge '{key}': {base_np is None} vs {feat_np is None}"
                )
            if base_np.shape != feat_np.shape:
                raise AssertionError(f"Feature-array shapes differ for edge '{key}': {base_np.shape} vs {feat_np.shape}")
            # numeric compare
            np.testing.assert_allclose(base_np, feat_np, rtol=rtol, atol=atol)

        # ADDRESS DICTS: keys must match, arrays comparable (possibly batched)
        base_addr_keys = set(base.port_dict.keys()) if base.port_dict is not None else set()
        for g in graphs[1:]:
            other_addr_keys = (
                set(g.hyper_edge_sets[key].port_dict.keys()) if g.hyper_edge_sets[key].port_dict is not None else set()
            )
            if base_addr_keys != other_addr_keys:
                raise AssertionError(f"Address dict keys differ for edge '{key}': {base_addr_keys} vs {other_addr_keys}")

        for ak in base_addr_keys:
            base_addr_np = np.array(base.port_dict[ak])
            for g in graphs[1:]:
                other_addr_np = np.array(g.hyper_edge_sets[key].port_dict[ak])
                if base_addr_np.shape != other_addr_np.shape:
                    raise AssertionError(
                        f"Address array shapes differ for edge '{key}' addr '{ak}': {base_addr_np.shape} vs {other_addr_np.shape}"
                    )
                np.testing.assert_allclose(base_addr_np, other_addr_np, rtol=rtol, atol=atol)

        # non_fictitious masks
        base_nf = np.array(base.non_fictitious) if base.non_fictitious is not None else None
        for g in graphs[1:]:
            other_nf = (
                np.array(g.hyper_edge_sets[key].non_fictitious) if g.hyper_edge_sets[key].non_fictitious is not None else None
            )
            if (base_nf is None) != (other_nf is None):
                raise AssertionError(f"Non-fictitious presence mismatch for edge '{key}'")
            if base_nf is not None:
                if base_nf.shape != other_nf.shape:
                    raise AssertionError(f"Non-fictitious shapes differ for edge '{key}': {base_nf.shape} vs {other_nf.shape}")
                np.testing.assert_allclose(base_nf, other_nf, rtol=rtol, atol=atol)
