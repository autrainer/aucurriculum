.. _scoring_functions:

Scoring Functions
=================

Scoring functions compute a difficulty score for each sample in a dataset.
Difficulty scores are converted into a difficulty ordering by ranking all examples by ascending sample difficulty.

.. tip::

   To create custom scoring functions, refer to the :ref:`custom scoring functions tutorial <tut_scoring_functions>`.

Such a difficulty ordering may be used to create a curriculum, a sequence of samples ordered by difficulty, which can be used in downstream tasks
to train a model in combination with a :ref:`pacing function <pacing_functions>`.

Most scoring functions obtain sample difficulty scores from a single training configuration.
`aucurriculum` allows for the automatic creation of ensemble scoring functions which may consist of multiple configurations
and obtain the final difficulty ordering by averaging the per-example difficulty scores across all atomic scoring functions.

.. note::
    `aucurriculum` currently supports scoring functions exclusively for multi-class classification tasks,
    both for obtaining sample difficulty scores and for curriculum-based training.

`aucurriculum` povides :ref:`model-based <scoring_functions_model_based>`,
:ref:`predefined <scoring_functions_predefined>`,
and :ref:`random <scoring_functions_random>` scoring functions to compute sample difficulty scores.

.. tip::
   All scoring function configurations contain placeholder values (indicated by :attr:`???`) that need to be replaced with the appropriate values.
   For more information on how to configure scoring functions, refer to the :ref:`quickstart guide <quickstart>`.

Curriculum Score Manager
------------------------

:class:`~aucurriculum.curricula.CurriculumScoreManager` manages the calculation of scoring functions in three steps:

1. :meth:`~aucurriculum.curricula.CurriculumScoreManager.preprocess`: Preprocess a scoring function configuration,
   creating one or more atomic scoring functions.
2. :meth:`~aucurriculum.curricula.CurriculumScoreManager.run`: Run a single atomic scoring function, producing a difficulty score for each sample.
3. :meth:`~aucurriculum.curricula.CurriculumScoreManager.postprocess`: Optionally combine multiple atomic scoring functions,
   creating a single difficulty score for each sample in a dataset.

.. autoclass:: aucurriculum.curricula.CurriculumScoreManager
   :members:


Abstract Scoring Function
-------------------------

All scoring functions inherit from the :class:`~aucurriculum.curricula.scoring.AbstractScore` class and implement the
:meth:`~aucurriculum.curricula.scoring.AbstractScore.run` method calculating the difficulty scores for each sample in the dataset.
:class:`~aucurriculum.curricula.scoring.AbstractScore` additionally provides common methods that are shared among most scoring functions.

.. note::
   Scoring functions can optionally override the :meth:`~aucurriculum.curricula.scoring.AbstractScore.preprocess` and
   :meth:`~aucurriculum.curricula.scoring.AbstractScore.postprocess` methods to perform additional operations before
   and after the scoring function is run, such as combining multiple atomic scoring functions in a different way than averaging.

.. autoclass:: aucurriculum.curricula.scoring.AbstractScore
   :members:


.. _scoring_functions_model_based:

Model-based Scoring Functions
-----------------------------

Model-based scoring functions obtain a difficulty score for each sample by leveraging trained models
and using, among others, training dynamics, model predictions, or losses to determine the difficulty of a sample.

Most model-based scoring functions require a trained model to compute the difficulty scores which is specified in the scoring function configuration
under the :attr:`run_name` parameter.
The :attr:`run_name` should be a run name or list of run names from which to load the models for scoring and should exist in the
:attr:`results_dir` and :attr:`experiment_id` set in the curriculum scoring configuration (:file:`conf/curriculum.yaml`) file.
It is also possible to specify (lists of) aggregated run names which are automatically resolved to the underlying runs,
effectively creating an ensemble scoring function.

.. autoclass:: aucurriculum.curricula.scoring.CELoss

   .. dropdown:: Default Configurations

      .. configurations::
         :subdir: curriculum/scoring
         :configs: CELoss

.. autoclass:: aucurriculum.curricula.scoring.CumulativeAccuracy

   .. dropdown:: Default Configurations

      .. configurations::
         :subdir: curriculum/scoring
         :configs: CumulativeAccuracy

.. autoclass:: aucurriculum.curricula.scoring.CVLoss

   .. dropdown:: Default Configurations

      .. configurations::
         :subdir: curriculum/scoring
         :configs: CVLoss

.. autoclass:: aucurriculum.curricula.scoring.FirstIteration

   .. dropdown:: Default Configurations

      .. configurations::
         :subdir: curriculum/scoring
         :configs: FirstIteration

.. autoclass:: aucurriculum.curricula.scoring.PredictionDepth

   .. dropdown:: Default Configurations

      .. configurations::
         :subdir: curriculum/scoring
         :configs: PredictionDepth

.. autoclass:: aucurriculum.curricula.scoring.TransferTeacher

   .. dropdown:: Default Configurations

      .. configurations::
         :subdir: curriculum/scoring
         :configs: TransferTeacher
         

.. _scoring_functions_predefined:

Predefined Scoring Functions
----------------------------

Predefined scoring functions determine the difficulty of a sample based on predefined criteria and are specified in a CSV file.

.. autoclass:: aucurriculum.curricula.scoring.Predefined

   .. dropdown:: Default Configurations

      .. configurations::
         :subdir: curriculum/scoring
         :configs: Predefined


.. _scoring_functions_random:

Random Scoring Functions
------------------------

Random scoring functions assign a random difficulty score to each sample in the dataset.

.. autoclass:: aucurriculum.curricula.scoring.Random

   .. dropdown:: Default Configurations

      .. configurations::
         :subdir: curriculum/scoring
         :configs: Random