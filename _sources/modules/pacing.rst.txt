.. _pacing_functions:

Pacing Functions
================

Pacing functions control the introduction of new samples into the curriculum-based training process
by determining the dataset size at each iteration.

.. tip::

   To create custom pacing functions, refer to the :ref:`custom pacing functions tutorial <tut_pacing_functions>`.

Based on the difficulty ordering provided by a :ref:`scoring function <scoring_functions>`,
new samples are introduced into the training process in ascending (or descending) order of difficulty.

.. note::
   `aucurriculum` introduces new samples into the training process after all currently available samples
   have been seen by the model at least once (i.e., once the training loader is exhausted).

   While the order of introducing new samples is determined by the scoring function,
   all currently available samples are shuffled before being introduced into the training process.


Curriculum Pace Manager
-----------------------

:class:`~aucurriculum.curricula.CurriculumPaceManager` manages the pacing of the curriculum by dynamically controlling the training loader,
shuffling the currently available dataset samples, and introducing new samples according to the
:ref:`curriculum training configuration <main_configuration_training>`.

.. autoclass:: aucurriculum.curricula.CurriculumPaceManager
   :members: cb_on_loader_exhausted, train_loader, shuffle_indices


Abstract Pacing Function
------------------------

All pacing functions inherit from the :class:`~aucurriculum.curricula.pacing.AbstractPace` class and implement the
:meth:`~aucurriculum.curricula.pacing.AbstractPace.get_dataset_size` method determining the size of the dataset to be used at a given iteration.

.. autoclass:: aucurriculum.curricula.pacing.AbstractPace
   :members:


Continuous Pacing Functions
---------------------------

Continuous pacing functions introduce new samples continuously, adding more samples after each iteration.

.. autoclass:: aucurriculum.curricula.pacing.Exponential

   .. dropdown:: Default Configurations

      .. configurations::
         :subdir: curriculum/pacing
         :configs: Exponential

.. autoclass:: aucurriculum.curricula.pacing.Logarithmic

   .. dropdown:: Default Configurations

      .. configurations::
         :subdir: curriculum/pacing
         :configs: Logarithmic

.. autoclass:: aucurriculum.curricula.pacing.Linear

   .. dropdown:: Default Configurations

      .. configurations::
         :subdir: curriculum/pacing
         :configs: Linear

.. autoclass:: aucurriculum.curricula.pacing.Quadratic

   .. dropdown:: Default Configurations

      .. configurations::
         :subdir: curriculum/pacing
         :configs: Quadratic

.. autoclass:: aucurriculum.curricula.pacing.Root

   .. dropdown:: Default Configurations

      .. configurations::
         :subdir: curriculum/pacing
         :configs: Root

.. autoclass:: aucurriculum.curricula.pacing.Polynomial

   .. dropdown:: Default Configurations

      No default configurations are provided for :class:`~aucurriculum.curricula.pacing.Polynomial`.


Discrete Pacing Functions
-------------------------

Discrete pacing functions introduce new samples at fixed intervals, such as after a set number of iterations.

.. autoclass:: aucurriculum.curricula.pacing.OneStep

   .. dropdown:: Default Configurations

      .. configurations::
         :subdir: curriculum/pacing
         :configs: OneStep