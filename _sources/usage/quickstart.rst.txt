.. _quickstart:

Quickstart
==========

The following quickstart guide provides a short introduction to `aucurriculum`
and the creation of simple curriculum scoring and training experiments.

.. tip::
   The quickstart example uses epoch-based training for the curriculum-based training,
   leading to a different number of training steps for each epoch, as the training subset size is reduced by the curriculum.
   To use a fixed number of training steps (which allows for an easier alignment of models trained with and without a curriculum),
   refer to the
   `autrainer step-based training documentation <https://autrainer.github.io/autrainer/usage/quickstart.html#training-duration-step-based-training>`_.


Training without a Curriculum
-----------------------------

To get started, create a new directory and navigate to it:

.. code-block:: extended

    mkdir aucurriculum_example && cd aucurriculum_example

Next, create a new empty `aucurriculum` project using the following :ref:`configuration management <cli_configuration_management>` CLI command:

.. code-block:: aucurriculum

    aucurriculum create --empty

Alternatively, use the following :ref:`configuration management <cli_wrapper_configuration_management>` CLI wrapper function:

.. code-block:: python

    import aucurriculum.cli

    aucurriculum.cli.create(empty=True)


This will create the :ref:`configuration directory structure <configuration_directories>`
and the :ref:`main training configuration <main_configuration>` (:file:`conf/config.yaml`) file with default values:

.. configurations::
   :configs: config
   :exact: 

Alongside the main training configuration file, the :ref:`main curriculum configuration <main_configuration_scoring>`
(:file:`conf/curriculum.yaml`) file to obtain sample difficulty ordering is created with default values:

.. configurations::
   :configs: curriculum
   :exact:


To train a model without a curriculum, run the following :ref:`training <cli_training>` CLI command:

.. code-block:: aucurriculum

    aucurriculum train

Alternatively, use the following :ref:`training <cli_wrapper_training>` CLI wrapper function:

.. code-block:: python

    aucurriculum.cli.train()

This will train the default :attr:`ToyFFNN` feed-forward neural network
(`autrainer.models.FFNN <https://autrainer.github.io/autrainer/modules/models.html#autrainer.models.FFNN>`_) on the default
:attr:`ToyTabular-C` classification dataset with tabular data
(`autrainer.datasets.ToyDataset <https://autrainer.github.io/autrainer/modules/datasets.html#autrainer.datasets.ToyDataset>`_)
and output the training results to the :file:`results/default/` directory.

Obtaining Difficulty Scores
---------------------------

After training, models can be used to obtain difficulty scores for samples in combination with :ref:`scoring functions <scoring_functions>`
and the :ref:`main curriculum configuration <main_configuration_scoring>` (:file:`conf/curriculum.yaml`) file.

Most scoring functions have a :attr:`run_name` parameter that specifies the run name or list of run names from which to load the models for scoring.
By default, this parameter is a placeholder (indicated by :attr:`???`) and has to be manually replaced with the run name of the trained model.

For example, create a local :class:`~aucurriculum.curricula.scoring.CELoss` scoring function configuration file
(:file:`conf/curriculum/scoring/CELoss.yaml`), replacing the :attr:`run_name` with the previously trained run:

.. literalinclude:: ../examples/quickstart/CELoss.yaml
   :language: yaml
   :caption: conf/curriculum/scoring/CELoss.yaml
   :linenos:

Next, replace the :attr:`curriculum/scoring` parameter in the main curriculum configuration file
(:file:`conf/curriculum.yaml`) with the scoring function ID:

.. literalinclude:: ../examples/quickstart/curriculum.yaml
   :language: yaml
   :caption: conf/curriculum.yaml
   :linenos:

To obtain the difficulty scores for the samples in the dataset, run the following :ref:`curriculum <cli_curriculum>` CLI command:

.. code-block:: aucurriculum

    aucurriculum curriculum

Alternatively, use the following :ref:`curriculum <cli_wrapper_curriculum>` CLI wrapper function:

.. code-block:: python

    aucurriculum.cli.curriculum()

Training with a Curriculum
--------------------------

After obtaining the difficulty scores, the :ref:`scoring function <scoring_functions>` can be used in combination with a
:ref:`pacing function <pacing_functions>` to create a curriculum for training.

For example, create a new main training configuration file (:file:`conf/curriculum_training.yaml`) with the following parameters:

.. literalinclude:: ../examples/quickstart/curriculum_training.yaml
   :language: yaml
   :caption: conf/curriculum_training.yaml
   :linenos:


Next, train a model with the scoring function and two different pacing functions using the following :ref:`training <cli_training>` CLI command:

.. code-block:: aucurriculum

    aucurriculum train -cn curriculum_training

Alternatively, use the following :ref:`training <cli_wrapper_training>` CLI wrapper function:

.. code-block:: python

    aucurriculum.cli.train(config_name="curriculum_training")

Finally, to compare the results of the run without a curriculum to the runs with a curriculum as well as averaged across pacing functions,
run the following :ref:`postprocessing <cli_postprocessing>` command:

.. code-block:: aucurriculum

    aucurriculum postprocess results default --aggregate curriculum.pacing

Alternatively, use the following :ref:`postprocessing <cli_wrapper_postprocessing>` CLI wrapper function:

.. code-block:: python

    aucurriculum.cli.postprocess(
        results_dir="results",
        experiment_id="default",
        aggregate=[["curriculum.pacing"]],
    )


Next Steps
----------

For more information on creating configurations, refer to the `autrainer quickstart <https://autrainer.github.io/autrainer/usage/quickstart.html>`_,
:ref:`Hydra configurations <hydra_configurations>`, as well as the `Hydra <https://hydra.cc/>`_ documentation.

To create custom scoring and pacing functions alongside configurations, refer to the :ref:`tutorials <tutorials>`.