.. _cli_wrapper:

CLI Wrapper
===========

`aucurriculum` provides an :attr:`aucurriculum.cli` CLI wrapper to programmatically manage the entire curriculum learning-based training process including
:ref:`configuration management <cli_wrapper_configuration_management>`,
:ref:`data preprocessing <cli_wrapper_preprocessing>`,
:ref:`model training <cli_wrapper_training>`,
:ref:`inference <cli_wrapper_inference>`,
:ref:`postprocessing <cli_wrapper_postprocessing>`,
and :ref:`curriculum <cli_wrapper_curriculum>` to obtain sample difficulty scores.

Wrapper functions are useful for integrating `aucurriculum` into custom scripts, jupyter notebooks, google colab notebooks, and other applications.

In addition to the CLI wrapper functions, `aucurriculum` provides a :ref:`CLI <cli_reference>` to manage configurations, data, training, inference,
and postprocessing from the command line with the same functionality as the CLI wrapper.

.. note::
    
   Both the CLI and the Python CLI wrapper serve as a curriculum learning extension to `autrainer`.
   For more information on the `autrainer` CLI wrapper, refer to the
   `autrainer CLI wrapper <https://autrainer.github.io/autrainer/usage/cli_wrapper.html>`_.


.. _cli_wrapper_aucurriculum:

aucurriculum
------------

.. argparse::
   :module: aucurriculum.core.scripts.cli
   :func: get_parser
   :prog: aucurriculum
   :nodefault:
   :noepilog:
   :nosubcommands:

   .. code-block:: aucli

      usage: aucurriculum [-h] [-v] <command> ...


.. _cli_wrapper_configuration_management:

Configuration Management
------------------------

To manage configurations, :ref:`aucurriculum create <cli_wrapper_aucurriculum_create>`, :ref:`aucurriculum list <cli_wrapper_aucurriculum_list>`,
and :ref:`aucurriculum show <cli_wrapper_aucurriculum_show>` allow for the creation of the project structure and the discovery
as well as saving of default configurations provided by `aucurriculum` and `autrainer`.

.. tip::
   
   Default configurations can be discovered both through the :ref:`CLI <cli_reference>`,
   the :ref:`CLI wrapper <cli_wrapper>`, and the respective module documentation.


.. _cli_wrapper_aucurriculum_create:

.. autofunction:: aucurriculum.cli.create

.. _cli_wrapper_aucurriculum_list:

.. autofunction:: aucurriculum.cli.list

.. _cli_wrapper_aucurriculum_show:

.. autofunction:: aucurriculum.cli.show


.. _cli_wrapper_preprocessing:

Preprocessing
-------------

To avoid race conditions when using `hydra_launcher_plugins <https://autrainer.github.io/autrainer/usage/hydra_configurations.html#launcher-plugins>`_
that may run multiple training jobs in parallel,
:ref:`aucurriculum fetch <cli_wrapper_aucurriculum_fetch>` and :ref:`aucurriculum preprocess <cli_wrapper_aucurriculum_preprocess>` allow for
downloading and `preprocessing <https://autrainer.github.io/autrainer/modules/transforms.html#preprocessing-transforms>`_
of `datasets <https://autrainer.github.io/autrainer/modules/datasets.html>`_ (and pretrained model states) before training.

Both commands are based on the :ref:`main configuration <main_configuration>` file (e.g. :file:`conf/config.yaml`),
such that the specified models and datasets are fetched and preprocessed accordingly.
If a model or dataset is already fetched or preprocessed, it will be skipped.

.. _cli_wrapper_aucurriculum_fetch:
.. _cli_wrapper_aucurriculum_preprocess:

For more information on fetching and preprocessing, refer to the
`autrainer preprocessing CLI wrapper <https://autrainer.github.io/autrainer/usage/cli_wrapper.html#preprocessing>`_.


.. _cli_wrapper_training:

Training
--------

Training is managed by :ref:`aucurriculum train <cli_wrapper_aucurriculum_train>`, which starts the training process
based on the :ref:`main configuration <main_configuration>` file (e.g. :file:`conf/config.yaml`).

.. _cli_wrapper_aucurriculum_train:

.. autofunction:: aucurriculum.cli.train


.. _cli_wrapper_inference:
.. _cli_wrapper_aucurriculum_inference:

Inference
---------

:ref:`aucurriculum inference <cli_wrapper_aucurriculum_inference>` allows for the (sliding window) inference of audio data using a trained model.
For more information on the inference process, refer to the
`autrainer inference CLI wrapper <https://autrainer.github.io/autrainer/usage/cli_wrapper.html#inference>`_.


.. _cli_wrapper_postprocessing:

Postprocessing
--------------

Postprocessing allows for the summarization, visualization, and aggregation of the training results using
:ref:`aucurriculum postprocess <cli_wrapper_aucurriculum_postprocess>`.
Several cleanup utilities are provided by :ref:`aucurriculum rm-failed <cli_wrapper_aucurriculum_rm_failed>`
and :ref:`aucurriculum rm-states <cli_wrapper_aucurriculum_rm_states>`.
Manual grouping of the training results can be done using :ref:`aucurriculum group <cli_wrapper_aucurriculum_group>`.


.. _cli_wrapper_aucurriculum_postprocess:

.. autofunction:: aucurriculum.cli.postprocess


.. _cli_wrapper_aucurriculum_rm_failed:
.. _cli_wrapper_aucurriculum_rm_states:
.. _cli_wrapper_aucurriculum_group:

For more information on the cleanup utilities and manual grouping, refer to the
`autrainer postprocessing CLI wrapper <https://autrainer.github.io/autrainer/usage/cli_wrapper.html#postprocessing>`_.

.. _cli_wrapper_curriculum:

Curriculum Scoring

Curriculum scoring is managed by :ref:`aucurriculum curriculum <cli_wrapper_aucurriculum_curriculum>`, which allows for the calculation of
sample difficulty scores based on the training results and :ref:`scoring functions <scoring_functions>`.

.. _cli_wrapper_aucurriculum_curriculum:

.. autofunction:: aucurriculum.cli.curriculum

