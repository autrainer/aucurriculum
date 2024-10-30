.. _contributing:

Contributing
============

Contributions are welcome!
If you would like to contribute to `aucurriculum`, please open an issue or a pull request on the `GitHub repository <https://github.com/autrainer/aucurriculum>`_.

Installation
------------

To install `aucurriculum`, first ensure that PyTorch (along with torchvision and torchaudio) version 2.0 or higher is installed.
For installation instructions, refer to the `PyTorch website <https://pytorch.org/get-started/locally/>`_.

It is recommended to install `aucurriculum` within a virtual environment.
To create a new virtual environment, refer to the `Python venv documentation <https://docs.python.org/3/library/venv.html>`_.

To set up `aucurriculum` for development, start by cloning the repository and then install the development dependencies with `poetry`:

.. code-block:: extended

   git clone git@github.com:autrainer/aucurriculum.git
   cd aucurriculum
   poetry install --all-extras


Conventions
-----------

We use `Ruff <https://docs.astral.sh/ruff/>`_ to enforce code style and formatting.
Common spelling errors are automatically corrected by `codespell <https://github.com/codespell-project/codespell>`_.

Linting, formatting, and spelling checks are run with `pre-commit <https://pre-commit.com/>`_.
To install the pre-commit hooks, run the following command:

.. code-block:: extended

   pre-commit install

Tests
-----

We use `pytest <https://docs.pytest.org/en/stable/>`_ for testing.
To run the tests, use the following command:

.. code-block:: extended

   pytest tests