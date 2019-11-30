====
Gräy
====

My take on agil.

Terminology
===========

Gräy task
           Each project has a potential count of unknown tasks. The amount of
           gräy tasks is estimated by the user. A gräy task takes the average
           time of known tasks.

Task
           A task is built from actions. An action is always an hour long. We
           always assume that a task has at least 4 actions.

Actions
           Actions are the things needed to complete a task. The user has to try
           to come up with actions that take roughly an hour.

Duration
           The time an action actually took in hours.

Factor
           The factor is the sum of durations diveded by the count of actions.

Formulas
           * factor   = sum(duration) / count(duration)

           * task[x]  = sum(duration[x])

           * estimate = (graey * avg(task) + sum(task)) * factor

           * done     = sum(duration)

TODO
====

* Add init db function, which can take an action_duration
