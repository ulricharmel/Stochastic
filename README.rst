======
Stochastic
======

**A**\pplication for fitting source models to **R**\adio **I**\nterferometric visibilites using **S**\tochastic **O**\ptimisation techniques

==============
Installation
==============
Installation from source_,
working directory where source is checked out

Do everything in virtual environment

.. code-block:: bash
    
    $ virtualenv -p python3.8 venv
    
    $ source venv/bin/activate
    
    $ venv
    
    $ git clone https://github.com/ulricharmel/Stochastic.git

    $ cd Stochastic

    $ git checkout dsi

    $ cd ..
 
    $ pip install -e Stochastic/
 
To run and see options

.. code-block:: bash

    $ stochastic --help 

    $ stochastic-prep --help

=======
License
=======

This project is licensed under the GNU General Public License v3.0 - see license_ for details.
