
.. image:: _static/logo_banner.png
    :alt: aucurriculum — A Curriculum Learning Toolkit for Deep Learning Tasks built on top of autrainer
    :align: center


aucurriculum
============

|pypi| |python_versions| |license|

A Curriculum Learning Toolkit for Deep Learning Tasks built on top of `autrainer <https://github.com/autrainer/autrainer>`_.

.. _installation:

Installation
------------

To install `aucurriculum`, first ensure that PyTorch (along with torchvision and torchaudio) version 2.0 or higher is installed.
For installation instructions, refer to the `PyTorch website <https://pytorch.org/get-started/locally/>`_.

It is recommended to install `aucurriculum` within a virtual environment.
To create a new virtual environment, refer to the `Python venv documentation <https://docs.python.org/3/library/venv.html>`_.

Next, install `aucurriculum` using `pip`.

.. code-block:: pip

   pip install aucurriculum

To install `aucurriculum` from source, refer to the :ref:`contribution guide <contributing>`.


Next Steps
----------

To get started using `aucurriculum`, the :ref:`quickstart guide <quickstart>` outlines the creation of a simple training configuration
and :ref:`tutorials` provide examples for implementing custom scoring and pacing functions including their configurations.

For a complete list of available CLI commands, refer to the :ref:`CLI reference <cli_reference>` or the :ref:`CLI wrapper <cli_wrapper>`.

.. |pypi| image:: https://img.shields.io/pypi/v/aucurriculum?logo=pypi&logoColor=b4befe&color=b4befe
   :target: https://pypi.org/project/aucurriculum/
   :alt: aucurriculum PyPI Version

.. |python_versions| image:: https://img.shields.io/pypi/pyversions/aucurriculum?logo=python&logoColor=b4befe&color=b4befe
   :target: https://pypi.org/project/aucurriculum/
   :alt: aucurriculum Python Versions

.. |license| image:: https://img.shields.io/badge/license-MIT-b4befe?logo=c
   :target: https://github.com/autrainer/aucurriculum/blob/main/LICENSE
   :alt: aucurriculum GitHub License