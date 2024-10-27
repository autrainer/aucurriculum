.. _core:

Core
====

Core provides various utilities and entry points for the _aucurriculum_ toolkit.


Entry Point
-----------

The main training entry point for `aucurriculum` serving as an overlay of
`autrainer.main <https://autrainer.github.io/autrainer/modules/core.html#entry-point>`_.

.. autofunction:: aucurriculum.main


.. _core_filters:

Filters
-------

To automatically filter out the placeholder scoring function or partial curricula, `aucurriculum` provides default filters which extend
the `hydra-filter-sweeper <https://github.com/autrainer/hydra-filter-sweeper/>`_ plugin.

.. autoclass:: aucurriculum.core.filters.FilterPlaceholderScoringFunction
   :members:

.. autoclass:: aucurriculum.core.filters.FilterPartialCurriculum
   :members:


Plotting
--------

Plotting provides a simple interface extending the `autrainer plotting <https://autrainer.github.io/autrainer/modules/core.html#plotting>`_
utilities to additionally plot the dynamic dataset size of the curriculum.


.. autoclass:: aucurriculum.core.plotting.CurriculumPlots
   :members:


Constants
---------

`aucurriculum` extends the default `autrainer.constants <https://autrainer.github.io/autrainer/modules/core.html#constants>`_ with additional
constants for the curriculum.

.. autoclass:: aucurriculum.core.constants.CurriculumConstants
   :members: