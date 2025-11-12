=============
Feature store
=============

.. currentmodule:: energnn.feature_store

This module provides the :class:`FeatureStoreClient` class, a turnkey Python client for interacting 
with an Feature Store server. It offers end-to-end management of configuration files, 
problem instances, and datasets: registering, retrieving, downloading, and remote storage 
via HTTP and a pluggable storage backend. 

This class implements the publication and consumption of problem-generation
artifacts (configuration files, :class:`energnn.problem.Problem` instances, :class:`energnn.problem.dataset.ProblemDataset`
collections) while preserving and verifying their metadata (hashes, versions, tags, timestamps) in the feature 
store database.


.. autoclass:: FeatureStoreClient
   :no-members:
   :show-inheritance:

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

    FeatureStoreClient.register_config
    FeatureStoreClient.get_configs_metadata
    FeatureStoreClient.get_config_metadata
    FeatureStoreClient.register_instance
    FeatureStoreClient.get_instances_metadata
    FeatureStoreClient.get_instance_metadata
    FeatureStoreClient.download_instance
    FeatureStoreClient.register_dataset
    FeatureStoreClient.get_datasets_metadata
    FeatureStoreClient.get_dataset_metadata
    FeatureStoreClient.download_dataset