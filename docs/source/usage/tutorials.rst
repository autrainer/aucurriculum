.. _tutorials:

Tutorials
=========

`aucurriculum` is designed to be flexible and extensible, allowing for the creation of custom ...

* :ref:`scoring functions <tut_scoring_functions>`
* :ref:`pacing functions <tut_pacing_functions>`

For each, a tutorial is provided below to demonstrate their implementation and configuration.

For the following tutorials, all Python files should be placed in the project root directory
and all configuration files should be placed in the corresponding subdirectories of the :file:`conf/` directory.


.. _tut_scoring_functions:

Custom Scoring Functions
------------------------

To create a custom :ref:`scoring function <scoring_functions>`, inherit from :class:`~aucurriculum.curricula.scoring.AbstractScore`
and implement the :meth:`~aucurriculum.curricula.scoring.AbstractScore.run` method.

For example, the following :ref:`model-based <scoring_functions_model_based>` scoring function determines the difficulty of each sample
by computing the probability (assuming higher probabilities indicate easier samples) of the most likely class
(regardless of the true class):

.. literalinclude:: ../examples/tutorials/probability_score.py
   :language: python
   :caption: probability_score.py
   :linenos:

Next, create a :file:`ProbabilityScore.yaml` configuration file for the scoring function in the :file:`conf/curriculum/scoring/` directory:

.. literalinclude:: ../examples/tutorials/ProbabilityScore.yaml
   :language: yaml
   :caption: conf/curriculum/scoring/ProbabilityScore.yaml
   :linenos:

The :attr:`id` should match the name of the configuration file.
The :attr:`_target_` should point to the custom scoring function class via a Python import path
(here assuming that the :file:`probability_score.py` file is in the root directory of the project).

The :attr:`run_name` should be a run name or list of run names from which to load the models for scoring.


.. _tut_pacing_functions:

Custom Pacing Functions
-----------------------

To create a custom :ref:`pacing function <pacing_functions>`, inherit from :class:`~aucurriculum.curricula.pacing.AbstractPace`
and implement the :meth:`~aucurriculum.curricula.pacing.AbstractPace.get_dataset_size` method.

For example, the following pacing function determines the dataset size at each iteration based on the convergence of the model,
adding a new discrete bucket of samples when the tracking metric does not improve for a specified number of iterations:

.. literalinclude:: ../examples/tutorials/discrete_convergence.py
   :language: python
   :caption: discrete_convergence.py
   :linenos:

Next, create a :file:`DiscreteConvergence.yaml` configuration file for the pacing function in the :file:`conf/curriculum/pacing/` directory:

.. literalinclude:: ../examples/tutorials/DiscreteConvergence.yaml
   :language: yaml
   :caption: conf/curriculum/pacing/DiscreteConvergence.yaml
   :linenos:

The :attr:`id` should match the name of the configuration file.
The :attr:`_target_` should point to the custom pacing function class via a Python import path
(here assuming that the :file:`discrete_convergence.py` file is in the root directory of the project).
The :attr:`patience` controls the number of iterations to wait for improvement before adding a new bucket of samples.
The :attr:`min_improvement` specifies the minimum improvement required to consider the model as having converged.
The :attr:`buckets` determines the number of discrete buckets the dataset is divided into.

Both the :attr:`initial_size` and :attr:`final_iteration` serve as placeholders (indicated by :attr:`???`)
and are automatically passed to the pacing function configuration
in the :ref:`main configuration <main_configuration>` file (e.g. :file:`conf/config.yaml`).