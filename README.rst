====
Gräy
====

My take on agil.

It will calculate/plot how much new work you discover while working on a
project. For example if you discover the same amount of work (or more) as
you get done, you will never finish. However if you discoer less work and
you get done, it will predict the total amount of work, using the gradient
(velocity) over the last 20 data-points. If there are less than 20 data-points
it will use the last third of the data-points.

You can also add your own prediction of how much work you will discover. I call
this gräy tasks.

Terminology
===========

Gräy task
           Each project has a potential count of unknown tasks. The amount of
           gräy tasks is estimated by the user. A gräy task takes the average
           time of known tasks.

Task
           A task is built from actions. An action is an hour long (by
           default).

Actions
           Actions are the things needed to complete a task. The user has to try
           to come up with actions that take roughly an hour (or another
           configured duration).

Duration
           The time an action actually took in hours.

Estimate
           The time estimated for actions, task or the whole project

Factor
           The factor is the sum of durations diveded by the count of actions.
