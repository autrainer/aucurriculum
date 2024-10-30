.. _hydra_configurations:

Hydra Configurations
====================

Analogous to `autrainer <https://autrainer.github.io/autrainer/usage/hydra_configurations.html>`_,
`aucurriculum` uses `Hydra <https://hydra.cc/docs/intro/>`_ to configure training experiments.
All configurations are stored in the :file:`conf/` directory and are defined as YAML files.


.. _main_configuration:

Main Configuration
------------------

`aucurriculum` extends the `autrainer main entry point <https://autrainer.github.io/autrainer/usage/hydra_configurations.html#main-configuration>`_
for curriculum training and adds an additional curriculum scoring entry point
for obtaining difficulty orderings from one or more :ref:`scoring functions <scoring_functions>`.


.. _main_configuration_training:

Curriculum Training
~~~~~~~~~~~~~~~~~~~

The main entry point for curriculum training is defined in the :file:`conf/config.yaml` file.

.. configurations::
   :configs: config
   :exact:

In addition to the `autrainer main configuration file <https://autrainer.github.io/autrainer/usage/hydra_configurations.html#default-config>`_
the following :attr:`hypdra/sweeper/params` parameters are defined:

* :attr:`curriculum`: The curriculum configuration. The following options are available:
  
  * :attr:`None`: No curriculum is used and training is performed on the full dataset. All other curriculum parameters are ignored.
  * :attr:`Curriculum`: A curriculum is used to train the model. New samples are introduced in order of increasing difficulty.
  * :attr:`AntiCurriculum`: An anti-curriculum is used to train the model. New samples are introduced in order of decreasing difficulty.

  :attr:`curriculum/sampling`: The sampling strategy of introducing new samples into the training set.
  The following strategies are available:
  
  * :attr:`Unbalanced`: Introduce new samples directly following the sample difficulty ordering provided by the
    :ref:`scoring function <scoring_functions>`.
  * :attr:`Balanced`: Introduce new samples in a class-balanced manner. New samples are introduced in a way
    that the class distribution of the training set remains balanced as long as possible, conceptually applying the difficulty ordering
    to each class separately.
  * :attr:`Original`: Introduce new samples with the same distribution as the original dataset. This is equivalent to
    introducing new samples in a class-balanced manner, but with the class distribution of the original dataset.

* :attr:`curriculum/scoring`: The ID of an already computed :ref:`scoring function <scoring_functions>` using the
  :ref:`curriculum scoring min configuration <main_configuration_scoring>` file.
* :attr:`curriculum/pacing`: The ID of a :ref:`pacing function <pacing_functions>`.
* :attr:`curriculum.pacing.initial_size`: The fraction of the dataset to be used in the first iteration.
  The remaining fraction is gradually introduced in subsequent iterations.
* :attr:`curriculum.pacing.final_iteration`: The fraction of iterations until the full dataset is used for training.

For each parameter, one or more values can be specified to sweep over different configurations.

If any of :attr:`curriculum`, :attr:`curriculum/sampling`, :attr:`curriculum/scoring`, or :attr:`curriculum/pacing`
are set to `None`, the run is filtered out and not executed.

.. _main_configuration_scoring:

Curriculum Scoring
~~~~~~~~~~~~~~~~~~

The main entry point for curriculum scoring is defined in the :file:`conf/curriculum.yaml` file.

.. configurations::
   :configs: curriculum
   :exact:

The following parameters are defined (alongside common attributes such as the :attr:`results_dir` or :attr:`experiment_id`):

* :attr:`curriculum/scoring` (under :attr:`hypdra/sweeper/params`): The :ref:`scoring function <scoring_functions>` ID to be calculated.
* :attr:`correlation`: A mapping of matrix names and list of scoring function IDs to calculate the correlation between scoring functions.
  :attr:`all` serves as a placeholder to include all scoring functions in the correlation matrix.

Some parameters of the main configuration file are outsourced to the :ref:`_aucurriculum_train_.yaml defaults <aucurriculum_defaults_train>`
and :ref:`_aucurriculum_score_.yaml defaults <aucurriculum_defaults_score>`
files in order to simplify the configurations.

For more information on configuring Hydra, see the `Hydra documentation <https://hydra.cc/docs/patterns/configuring_experiments/>`_.


.. tip::
   
   Analogous to `autrainer configurations <https://autrainer.github.io/autrainer/usage/hydra_configurations.html#main-configuration>`_,
   different files can be used as the main entry point for training and scoring experiments using
   the :attr:`-cn/--config-name` argument for the :ref:`aucurriculum train <cli_training>` CLI command:

   .. code-block:: aucurriculum
   
      aucurriculum train -cn some_other_config

   Alternatively, use the :attr:`config_name` parameter for the :meth:`~aucurriculum.cli.train` CLI wrapper function:

   .. code-block:: python

      aucurriculum.cli.train(config_name="some_other_config")
   
   For more information on command line flags, see
   `Hydra's command line flags documentation <https://hydra.cc/docs/advanced/hydra-command-line-flags/>`_.


.. _configuration_directories:

Configuration Directories
-------------------------

In addition to the
`autrainer configuration directories <https://autrainer.github.io/autrainer/usage/hydra_configurations.html#configuration-directories>`_,
the following configuration subdirectories are available:

* :file:`conf/curriculum/`
* :file:`conf/curriculum/sampling/`
* :file:`conf/curriculum/scoring/`
* :file:`conf/curriculum/pacing/`


.. _creating_configurations:
.. _shorthand_syntax:
.. _interpolation_syntax:
.. _defaults_list:
.. _optional_defaults_list:

Configurations
--------------

Analogous to `autrainer`, `aucurriculum` provides default configurations for scoring functions, pacing functions, etc. that can be
used out of the box without creating custom configurations.

For more information on creating, discovering, and using configurations, or special syntaxes and overriding defaults, refer to the
`autrainer configuration creation <https://autrainer.github.io/autrainer/usage/hydra_configurations.html#creating-configurations>`_ documentation.


.. _aucurriculum_defaults:

aucurriculum Defaults
---------------------
Both the :file:`_aucurriculum_train_.yaml` and :file:`_aucurriculum_score_.yaml` files contain further default configurations
to simplify the :ref:`main configuration <main_configuration>` files.

.. tip::

   Any global default parameter can be overridden in the :ref:`main configuration <main_configuration>` file by redefining it.

.. _aucurriculum_defaults_train:

.. configurations::
   :configs: _aucurriculum_train_
   :exact:

.. _aucurriculum_defaults_score:

.. configurations::
   :configs: _aucurriculum_score_
   :exact:


.. _hydra_plugins:

Hydra Plugins
-------------

By default, `aucurriculum` uses the `hydra-filter-sweeper <https://github.com/autrainer/hydra-filter-sweeper>`_
plugin to sweep over hyperparameter configurations which are defined in the :ref:`aucurriculum defaults <aucurriculum_defaults>`
and implemented in the :ref:`core module <core_filters>`.
This plugin allows to specify a list of :attr:`filters` in the configuration file to filter out unwanted hyperparameter combinations.

To perform more complex hyperparameter sweeps, different sweeper or launcher plugins can be used.
For more information on plugins, refer to the
`autrainer plugins <https://autrainer.github.io/autrainer/usage/hydra_configurations.html#hydra-plugings>`_ documentation.

.. note::
   
   Custom sweeper and launcher plugins are not yet supported for curriculum scoring.