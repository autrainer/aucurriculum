.. _cli_reference:

CLI Reference
=============

`aucurriculum` provides a command line interface (CLI) to manage the entire curriculum learning-based training process including
:ref:`configuration management <cli_configuration_management>`,
:ref:`data preprocessing <cli_preprocessing>`,
:ref:`model training <cli_training>`,
:ref:`inference <cli_inference>`,
:ref:`postprocessing <cli_postprocessing>`,
and :ref:`curriculum <cli_curriculum>` to obtain sample difficulty scores.

In addition to the CLI, `aucurriculum` provides a :ref:`Python CLI wrapper <cli_wrapper>` to manage configurations, data, training, inference,
postprocessing, and curriculum sample difficulty calculation programmatically with the same functionality as the CLI.

.. note::
    
   Both the CLI and the Python CLI wrapper serve as a curriculum learning extension to `autrainer`.
   For more information on the `autrainer` CLI, refer to the
   `autrainer CLI reference <https://autrainer.github.io/autrainer/usage/cli_reference.html>`_.


.. _cli_aucurriculum:

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


.. _cli_configuration_management:

Configuration Management
------------------------

To manage configurations, :ref:`aucurriculum create <cli_aucurriculum_create>`, :ref:`aucurriculum list <cli_aucurriculum_list>`,
and :ref:`aucurriculum show <cli_aucurriculum_show>` allow for the creation of the project structure and the discovery
as well as saving of default configurations provided by `aucurriculum` and `autrainer`.

.. tip::
   
   Default configurations can be discovered both through the :ref:`CLI <cli_reference>`,
   the :ref:`CLI wrapper <cli_wrapper>`, and the respective module documentation.


.. _cli_aucurriculum_create:

aucurriculum create
~~~~~~~~~~~~~~~~~~~

.. argparse::
   :module: aucurriculum.core.scripts.cli
   :func: get_parser
   :prog: aucurriculum
   :nodefault:
   :noepilog:
   :path: create

   .. code-block:: aucli

      usage: aucurriculum create [-h] [-e] [-a] [-f] [directories ...]


.. _cli_aucurriculum_list:

aucurriculum list
~~~~~~~~~~~~~~~~~

.. argparse::
   :module: aucurriculum.core.scripts.cli
   :func: get_parser
   :prog: aucurriculum
   :nodefault:
   :noepilog:
   :path: list

   .. code-block:: aucli

      usage: aucurriculum list [-h] [-l] [-g] [-p P] directory


.. _cli_aucurriculum_show:

aucurriculum show
~~~~~~~~~~~~~~~~~

.. argparse::
   :module: aucurriculum.core.scripts.cli
   :func: get_parser
   :prog: aucurriculum
   :nodefault:
   :noepilog:
   :path: show

   .. code-block:: aucli

      usage: aucurriculum show [-h] [-s] [-f] directory config


.. _cli_preprocessing:

Preprocessing
-------------

To avoid race conditions when using `hydra_launcher_plugins <https://autrainer.github.io/autrainer/usage/hydra_configurations.html#launcher-plugins>`_
that may run multiple training jobs in parallel,
:ref:`aucurriculum fetch <cli_aucurriculum_fetch>` and :ref:`aucurriculum preprocess <cli_aucurriculum_preprocess>` allow for
downloading and `preprocessing <https://autrainer.github.io/autrainer/modules/transforms.html#preprocessing-transforms>`_
of `datasets <https://autrainer.github.io/autrainer/modules/datasets.html>`_ (and pretrained model states) before training.

Both commands are based on the :ref:`main configuration <main_configuration>` file (e.g. :file:`conf/config.yaml`),
such that the specified models and datasets are fetched and preprocessed accordingly.
If a model or dataset is already fetched or preprocessed, it will be skipped.

.. _cli_aucurriculum_fetch:
.. _cli_aucurriculum_preprocess:

For more information on fetching and preprocessing, refer to the
`autrainer preprocessing CLI reference <https://autrainer.github.io/autrainer/usage/cli_reference.html#preprocessing>`_.


.. _cli_training:

Training
--------

Training is managed by :ref:`aucurriculum train <cli_aucurriculum_train>`, which starts the training process
based on the :ref:`main configuration <main_configuration>` file (e.g. :file:`conf/config.yaml`).

.. _cli_aucurriculum_train:

aucurriculum train
~~~~~~~~~~~~~~~~~~

.. argparse::
   :module: aucurriculum.core.scripts.cli
   :func: get_parser
   :prog: aucurriculum
   :nodefault:
   :noepilog:
   :path: train

   .. code-block:: aucli

      usage: aucurriculum train [-h]


.. _cli_inference:
.. _cli_aucurriculum_inference:

Inference
---------

:ref:`aucurriculum inference <cli_aucurriculum_inference>` allows for the (sliding window) inference of audio data using a trained model.
For more information on the inference process, refer to the
`autrainer inference CLI reference <https://autrainer.github.io/autrainer/usage/cli_reference.html#inference>`_.


.. _cli_postprocessing:

Postprocessing
--------------

Postprocessing allows for the summarization, visualization, and aggregation of the training results using :ref:`aucurriculum postprocess <cli_aucurriculum_postprocess>`.
Several cleanup utilities are provided by :ref:`aucurriculum rm-failed <cli_aucurriculum_rm_failed>` and :ref:`aucurriculum rm-states <cli_aucurriculum_rm_states>`.
Manual grouping of the training results can be done using :ref:`aucurriculum group <cli_aucurriculum_group>`.


.. _cli_aucurriculum_postprocess:

aucurriculum postprocess
~~~~~~~~~~~~~~~~~~~~~~~~

.. argparse::
   :module: aucurriculum.core.scripts.cli
   :func: get_parser
   :prog: aucurriculum
   :nodefault:
   :noepilog:
   :path: postprocess

   .. code-block:: aucli

      usage: aucurriculum postprocess [-h] [-m N] [-a A [A ...]] results_dir experiment_id


.. _cli_aucurriculum_rm_failed:
.. _cli_aucurriculum_rm_states:
.. _cli_aucurriculum_group:

For more information on the cleanup utilities and manual grouping, refer to the
`autrainer postprocessing CLI reference <https://autrainer.github.io/autrainer/usage/cli_reference.html#postprocessing>`_.

.. _cli_curriculum:

Curriculum Scoring

Curriculum scoring is managed by :ref:`aucurriculum curriculum <cli_aucurriculum_curriculum>`, which allows for the calculation of
sample difficulty scores based on the training results and :ref:`scoring functions <scoring_functions>`.

.. _cli_aucurriculum_curriculum:

aucurriculum curriculum
~~~~~~~~~~~~~~~~~~~~~~~

.. argparse::
   :module: aucurriculum.core.scripts.cli
   :func: get_parser
   :prog: aucurriculum
   :nodefault:
   :noepilog:
   :path: curriculum

   .. code-block:: aucli

      usage: aucurriculum curriculum -cn curriculum.yaml

