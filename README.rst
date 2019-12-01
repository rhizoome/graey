====
Gräy
====

My take on agil.

It calculates/plots how much new work you discover while working on
a project. For example if you discover the same amount of work (or more)
as you get done, you will never finish. However if you discover less work
than you get done, gräy will predict the total amount of work, using the slope
(velocity) over the last 20 data-points. If there are less than 20 data-points
it will use the last third of the data-points.

You can also add your own prediction of how much work you will discover. I call
this gräy tasks.

Gräy can only work if you update gräy as you complete actions or you discover
new actions.

.. raw:: html

    <img alt="plot" src="https://1042.ch/42/ganwell/graey.png" width="70%">

Terminology
===========

Gräy task
           A project can have a potential count of unknown tasks. The amount of
           gräy tasks is estimated by the user. A gräy task takes the average
           time of known tasks.

Task
           A task is built from actions. An action is an hour long (by
           default).

Actions
           Gräy works best if actions have a uniform duration and tasks have
           a uniform count of actions. Gräy tries to deal with variability
           by taking averages of the values needed for projections and
           predictions.

Duration
           The time an actions, task or the whole project actually took in hours.

Estimate
           The time estimated for actions, task or the whole project.

Projection
           The projection is updated by the real duration when an action is done.
           It is also corrected by the factor.

Factor
           The factor is the sum of durations diveded by the sum of estimates.

Prediction
           The prediction or trend is based on the corrected projection. It is
           the total time project will take if velocity stays the same. The last
           20 data-points are used for the prediction.

Velocity
           The proportion of completed work to newly discovered work.
           It is the slope of the plotted curve.

Work/Effort
           The sum of time assosiated with the actions.

Features
========

.. code-block:: text

   $> gry
   Usage: gry [OPTIONS] COMMAND [ARGS]...
   
   Options:
     --help  Show this message and exit.
   
   Commands:
     add    add ACTION to TASK
     csv    output as csv (projection, done)
     del    delete an ACTION by id (see show)
     done   complete an ACTION by id (DURATION in HHMM)
     est    set default ESTIMATE
     gry    set graey COUNT
     merge  fix database after automatic or manual merge
     plot   display plot
     save   save plot as html
     show   show open tasks
     stats  display stats
     tasks  show known tasks

html
----

Example_

.. _Example: http://1042.ch/42/ganwell/graey.html

Tasks
-----

.. code-block:: text

   $> gry tasks
    ─────────────────────────────────────────────────────────────────────────────────
     task   projection   projection (corr)   remaining   remaining (corr)   estimate
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        b         5.00                9.20        3.00               7.20       5.00
    ─────────────────────────────────────────────────────────────────────────────────
        a        10.00               10.00        0.00               0.00       4.00
    ─────────────────────────────────────────────────────────────────────────────────
   average projection:     9.60h    |    average actions:     4.00

Show
----

The slowest sandwhich maker in the world:

.. code-block:: text

   $> gry show
    ───────────────────────────────────────────────────
     id            task              action   estimate
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      1   make sandwich           get bread        1.0
    ───────────────────────────────────────────────────
      2   make sandwich         slice bread        1.0
    ───────────────────────────────────────────────────
      3   make sandwich     get ingredients        1.0
    ───────────────────────────────────────────────────
      4   make sandwich   layer ingredients        1.0
    ───────────────────────────────────────────────────
      5   make sandwich                 eat        1.0
    ───────────────────────────────────────────────────
     gräy: 0   |   default estimate: 1.0

Stats
-----

.. code-block:: text

   $> gry stats
   actions:                       8
   actions (done):                8
   actions (open):                0
   tasks:                         2
   tasks (done):                  2
   tasks (open):                  0
   tasks (gräy):                  0
   tasks (avg. actions):          4.00
   prediction data-points:        6
   projection:                   17.00h
   projection (corrected):       17.00h
   projection (predicted):       17.00h
   tasks (avg. projection):       8.50h
   estimate:                      8.00h
   correction factor:             2.12h
   done:                         17.00h
   remaining:                     0.00h
   remaining (corrected):         0.00h
   remaining (predicted):         0.00h

TODO
====

* I guess the fixed count of 20 prediction data-points is a problem for a very
  large project. I'm not sure if an option to plot and stats, is enough to fix
  the problem.

* merge
