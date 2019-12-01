import codecs
import math
import os
from collections import OrderedDict, namedtuple
from datetime import datetime
from subprocess import CalledProcessError, check_output
from uuid import uuid4

import click
import pytz
import termtables as tt
from pyrsistent import PRecord, field, pmap

try:
    import ujson as json
except ModuleNotFoundError:
    import json


matplotlib = None
plt = None
mpld3 = None
np = None


# Lazy loading of numpy and matplotlib improves startup time by a factor of 0.3
def import_np():
    global np
    import numpy as np


def import_plt():
    global matplotlib
    global plt
    global mpld3
    import matplotlib
    import matplotlib.pyplot as plt
    import mpld3

    import_np()


max_trend = 20


def invert(dict_):
    ret = {}
    for key in dict_.keys():
        value = dict_[key]
        ret[value] = key
    return ret


Meta = namedtuple("Meta", ("timestamp", "cmd"))
Add = namedtuple("Add", ("uuid", "task", "action", "estimate"))
Del = namedtuple("Del", ("uuid"))
Done = namedtuple("Done", ("uuid", "duration"))
Gry = namedtuple("Gry", ("count"))
Est = namedtuple("Est", ("estimate"))
cmds = {"add": Add, "del": Del, "done": Done, "gry": Gry, "est": Est}
cmds_inv = invert(cmds)


class Estimate(PRecord):
    open = field(initial=0)
    done = field(initial=0)
    projection = field(initial=0.0)
    estimate = field(initial=0.0)
    duration = field(initial=0.0)

    @property
    def all(self):
        return self.done + self.open

    def __repr__(self):
        return (
            f"est: {self.estimate:.2f}({self.all}) "
            f"proj: {self.projection:.2f} "
            f"dur: {self.duration:.2f}({self.done})"
        )


class Task(Estimate):
    pass


default_task = Task()


class State(Estimate):
    graey = field(initial=0)
    actions = field(initial=pmap())
    tasks = field(initial=pmap())
    default_est = field(initial=1.0)


db_error = "graey.db does not exist"
minute_error = """a duration has the format HHMM, the minute part must always be lower than 60

Correct: 20, 100, 0120, 120, 230, 0359
Incorrect: 70, 170, 199"""
merge_help = """
Fix database after automatic or manual merge. If you want can also concat all versions of the database:

    \b
    cat graey.db.BASE graey.db.LOCAL graey.db.REMOTE > graey.db
    gry merge

The entries will be sorted by timestamp and deduplicated.
""".strip()


@click.group()
def main():
    pass


@click.command(help="add ACTION to TASK")
@click.argument("TASK")
@click.option(
    "--estimate",
    "-e",
    type=click.INT,
    default=None,
    help="Estimate of the action (HHMM)",
)
@click.argument("ACTION", nargs=-1)
def add(task, estimate, action):
    if estimate is None:
        estimate = get_default_est()
    else:
        estimate = duration_to_hours(estimate)
    action = " ".join(action)
    with codecs.open("graey.db", "a+") as f:
        s = serialize(f, Add(str(uuid4()), task, action, estimate))
        f.write(f"{s}\n")
    print(f"added action {task} {action} with estimate {estimate:8.2f}h")


main.add_command(add)


@click.command(help="show open tasks")
@click.argument("TASK", nargs=-1)
def show(task):
    filter = None
    if task:
        filter = task[0]
    count, table = get_table()
    table = [
        (n, c.task, c.action, c.estimate)
        for n, c in table
        if filter is None or c.task == filter
    ]
    if table:
        print(
            tt.to_string(
                table,
                header=["id", "task", "action", "estimate"],
                style=tt.styles.booktabs,
                alignment="r",
            )
        )
        state = None
        for state in get_states():
            pass
        if state:
            print(f"  gräy: {count}   |   default estimate: {state.default_est}")
        else:
            print(f"  gräy: {count}")
    else:
        print("There are no open actions")


main.add_command(show)


@click.command("del", help="delete an ACTION by id (see show)")
@click.argument("ACTION", nargs=1, type=click.INT)
def delete(action):
    _, table = get_table()
    cmd = table[action - 1][1]
    with codecs.open("graey.db", "a+") as f:
        s = serialize(f, Del(cmd.uuid))
        f.write(f"{s}\n")
    print(f"marked action {action}: {cmd.task} {cmd.action} deleted")


main.add_command(delete)


@click.command(help="complete an ACTION by id (DURATION in HHMM)")
@click.argument("ACTION", nargs=1, type=click.INT)
@click.argument("DURATION", nargs=1, type=click.INT)
def done(action, duration):
    _, table = get_table()
    try:
        cmd = table[action - 1][1]
    except IndexError:
        raise click.ClickException(f"there is no task {action}")
    float_hours = duration_to_hours(duration)
    with codecs.open("graey.db", "a+") as f:
        s = serialize(f, Done(cmd.uuid, float_hours))
        f.write(f"{s}\n")
    print(
        f"marked action {action}: {cmd.task} {cmd.action} done duration: {float_hours:8.2f}h"
    )


main.add_command(done)


@click.command(help="show known tasks")
def tasks():
    state = None
    for state in get_states():
        pass
    if not state:
        return
    table = []
    tasks = state.tasks
    if not tasks:
        return
    factor = calc_factor(state)
    for idx in tasks.keys():
        task = tasks[idx]
        proj = correct(task, factor)
        table.append(
            (
                idx,
                fmt(task.projection),
                fmt(proj),
                fmt(task.projection - task.duration),
                fmt(proj - task.duration),
                fmt(task.estimate),
            )
        )
    print(
        tt.to_string(
            table,
            alignment="r",
            header=[
                "task",
                "projection",
                "projection (corr)",
                "remaining",
                "remaining (corr)",
                "estimate",
            ],
            style=tt.styles.booktabs,
        )
    )
    factor = calc_factor(state)
    avg_actions, avg_projection = avg_task_projection(state, factor)
    print(
        f"average projection: {avg_projection:8.2f}h    |"
        f"    average actions: {avg_actions:8.2f}"
    )


main.add_command(tasks)


@click.command("gry", help="set graey COUNT")
@click.argument("COUNT", nargs=1, type=click.INT)
def gry(count):
    with codecs.open("graey.db", "a+") as f:
        s = serialize(f, Gry(count))
        f.write(f"{s}\n")
    print(f"set gräy count to {count}")


main.add_command(gry)


@click.command("est", help="set default ESTIMATE")
@click.argument("ESTIMATE", nargs=1, type=click.INT)
def est(estimate):
    estimate = duration_to_hours(estimate)
    with codecs.open("graey.db", "a+") as f:
        s = serialize(f, Est(estimate))
        f.write(f"{s}\n")
    print(f"set estimate to {estimate}")


main.add_command(est)


@click.command(help="display stats")
def stats():
    states = list(get_states())
    calc = [calculate(state) for state in states]
    last = states[-1]
    lastc = calc[-1]
    tasks = 0
    tasks_open = 0
    tasks_done = 0
    for task in last.tasks.values():
        tasks += 1
        if task.open:
            tasks_open += 1
        else:
            tasks_done += 1
    part, start, end, grad = prediction_points(calc)
    x0 = start[0]
    y0 = start[0] - start[1]
    x1 = end[0]
    y1 = end[0] - end[1]

    # Because of prolonged disuse my algebra was broken

    # Given: P(x0, y0), P(x1, y1), f(x) = x * m + b

    # y0 = x0 * m + b
    # y1 = x1 * m + b

    # b = y0 - x0 * m
    # b = y1 - x1 * m

    # y0 - x0 * m = y1 - x1 * m           | - y0
    # - (x0 * m) = y1 - y0 - x1 * m       | + (x1 * m)
    # x1 * m - x0 * m = y1 - y0
    # m (x1 - x0) = y1 - y0               | / (x1 - x0)
    # m = (y1 - y0) / (x1 - x0)

    # y0 = x0 * m + b                     | - (x0 * m)
    # y0 - x0 * m = b
    # b = y0 - x0 * m
    # b = y0 - x0 * ((y1 - y0) / (x1 - x0))
    try:
        m = (y1 - y0) / (x1 - x0)
        b = y0 - x0 * ((y1 - y0) / (x1 - x0))
        pred = -(b / m)
        rem_pred = pred - last.duration
    except ZeroDivisionError:
        pred = None
        rem_pred = None
    rem = last.projection - last.duration
    rem_corr = lastc[0] - last.duration
    factor = calc_factor(last)
    avg_actions, avg_projection = avg_task_projection(last, factor)
    print(f"actions:                {last.all:8d}")
    print(f"actions (done):         {last.done:8d}")
    print(f"actions (open):         {last.open:8d}")
    print(f"tasks:                  {tasks:8d}")
    print(f"tasks (done):           {tasks_done:8d}")
    print(f"tasks (open):           {tasks_open:8d}")
    print(f"tasks (gräy):           {last.graey:8d}")
    print(f"tasks (avg. actions):      {avg_actions:8.2f}")
    print(f"prediction data-points: {part:8d}")
    print(f"projection:                {last.projection:8.2f}h")
    print(f"projection (corrected):    {lastc[0]:8.2f}h")
    if pred is None:
        print(f"projection (predicted):     (undef)h")
    else:
        print(f"projection (predicted):    {pred:8.2f}h")
    print(f"tasks (avg. projection):   {avg_projection:8.2f}h")
    print(f"estimate:                  {last.estimate:8.2f}h")
    print(f"correction factor:         {factor:8.2f}h")
    print(f"done:                      {last.duration:8.2f}h")
    print(f"remaining:                 {rem:8.2f}h")
    print(f"remaining (corrected):     {rem_corr:8.2f}h")
    if rem_pred is None:
        print(f"remaining (predicted):      (undef)h")
    else:
        print(f"remaining (predicted):     {rem_pred:8.2f}h")


main.add_command(stats)


@click.command(help="display plot")
def plot():
    import_plt()
    matplotlib.use("TkAgg")
    fig, ax = do_plot()
    try:
        out = check_output(["xrdb", "-query"])
        for line in out.decode("UTF-8").splitlines():
            if line.startswith("Xft.dpi"):
                splt = [x for x in line.split() if x]
                fig.set_dpi(int(splt[1]))
    except CalledProcessError:
        pass
    plt.show()


main.add_command(plot)


@click.command(help="save plot")
def save():
    import_plt()
    fig, ax = do_plot()
    mpld3.save_html(fig, "graey.html")
    print("saved plot to graey.html")


main.add_command(save)


@click.command(help="output as csv (projection, done)")
def csv():
    for state in get_states():
        prj, done = calculate(state)
        print(prj, done)


main.add_command(csv)


@click.command(
    short_help="fix database after automatic or manual merge", help=merge_help
)
def merge():
    raise click.ClickException("not implemented")


main.add_command(merge)


def fmt(value):
    return f"{value:8.2f}"


def get_default_est():
    projection = 1.0
    if os.path.exists("graey.db"):
        with codecs.open("graey.db", "r") as f:
            for line in f:
                meta, cmd = deserialize(line)
                if meta.cmd == "est":
                    projection = cmd.estimate
    return projection


def duration_to_hours(duration):
    hours = duration // 100
    minutes = duration - (hours * 100)
    if minutes >= 60:
        raise click.ClickException(minute_error)
    return (60 * hours + minutes) / 60


def prediction_points(calc):
    steps = len(calc)
    part = min(steps // 3, max_trend)
    if part < 4:
        part = min(steps, 4)
    start = calc[-part]
    end = calc[-1]
    grad = ((end[0] - start[0]) / part, (end[1] - start[1]) / part)
    return (part, start, end, grad)


def plot_effort(ax, calc):
    xs = []
    ys = []
    for x, y in calc:
        xs.append(x)
        ys.append(x - y)
    ax.plot(xs, ys, zorder=10)
    ax.set_xlabel("projected effort (hours)")
    ax.set_ylabel("open effort (hours)")


def plot_velocity(ax, calc):
    part, start, end, grad = prediction_points(calc)
    trend = max(part, 20)
    xs = np.linspace(start[0], start[0] + grad[0] * trend, trend)
    ys = np.linspace(start[1], start[1] + grad[1] * trend, trend)
    ds = xs - ys
    try:
        zero = np.where(np.diff(np.sign(ds)))[0][0] + 2
    except IndexError:
        zero = trend
    ax.plot(xs[:zero], ds[:zero], zorder=1)
    return part


def do_plot():
    states = list(get_states())
    calc = [calculate(state) for state in states]
    fig, ax = plt.subplots()
    plot_effort(ax, calc)
    part = plot_velocity(ax, calc)
    ax.legend(["data", f"trend ({part} data-points)"])
    lim = [ax.get_xlim(), ax.get_ylim()]
    # Transpose list
    lim = list(map(list, zip(*lim)))
    lim[0] = min(*lim[0])
    lim[1] = max(*lim[1])
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    plt.title("Effort proportion")
    return fig, ax


def get_states():
    state = State()
    if os.path.exists("graey.db"):
        with codecs.open("graey.db", "r") as f:
            for line in f:
                line = deserialize(line)
                state = update_state(state, line)
                yield state
    else:
        nodb()


def avg_task_projection(state, factor):
    tasks = state.tasks
    if tasks:
        projection = 0
        avg_actions = sum([x.all for x in tasks.values()]) / len(tasks)
        fill_actions = math.ceil(avg_actions)
        # fill_actions = max(4, avg_actions)  # No sure if I want this
        for task in tasks.values():
            all = task.all
            unknown = 0
            if all < fill_actions and task.open:
                unknown = max(0, fill_actions - all)
            projection += (
                unknown * state.default_est * factor
                + (task.projection - task.duration) * factor
                + task.duration
            )

        return avg_actions, projection / len(tasks)
    else:
        return 4, state.default_est * 4


def calc_factor(state):
    if state.duration and state.estimate:
        return state.duration / state.estimate
    else:
        return 1


def correct(item, factor):
    return (item.projection - item.duration) * factor + item.duration


def calculate(state):
    factor = calc_factor(state)
    _, avg_projection = avg_task_projection(state, factor)
    proj = correct(state, factor) + state.graey * avg_projection
    done = state.duration
    assert proj >= done
    return proj, done


def update_state(state, line):
    actions = state.actions
    tasks = state.tasks
    meta, cmd = line
    if meta.cmd == "add":
        actions = actions.set(cmd.uuid, cmd)
        task = tasks.get(cmd.task, default_task)
        task = task.set(
            open=task.open + 1,
            projection=task.projection + cmd.estimate,
            estimate=task.estimate + cmd.estimate,
        )
        tasks = tasks.set(cmd.task, task)
        return state.set(
            actions=actions,
            tasks=tasks,
            open=state.open + 1,
            projection=state.projection + cmd.estimate,
        )
    if meta.cmd == "gry":
        return state.set(graey=cmd.count)
    if meta.cmd == "est":
        return state.set(default_est=cmd.estimate)
    key = actions[cmd.uuid].task
    task = tasks[key]
    action = actions.get(cmd.uuid)
    if meta.cmd == "del":
        actions = actions.remove(cmd.uuid)
        task = task.set(
            open=task.open - 1,
            projection=task.projection - action.estimate,
            estimate=task.estimate - action.estimate,
        )
        tasks = tasks.set(key, task)
        return state.set(
            actions=actions,
            tasks=tasks,
            open=state.open - 1,
            projection=state.projection - action.estimate,
        )
    if meta.cmd == "done":
        actions = actions.remove(cmd.uuid)
        task = task.set(
            open=task.open - 1,
            done=task.done + 1,
            projection=task.projection + cmd.duration - action.estimate,
            estimate=state.estimate + action.estimate,
            duration=task.duration + cmd.duration,
        )
        tasks = tasks.set(key, task)
        return state.set(
            open=state.open - 1,
            done=state.done + 1,
            actions=actions,
            tasks=tasks,
            projection=state.projection + cmd.duration - action.estimate,
            estimate=state.estimate + action.estimate,
            duration=state.duration + cmd.duration,
        )
    assert False


def get_table():
    state = OrderedDict()
    count = 0
    if os.path.exists("graey.db"):
        with codecs.open("graey.db", "r") as f:
            for line in f:
                meta, cmd = deserialize(line)
                if meta[1] == "add":
                    state[cmd.uuid] = cmd
                elif meta[1] == "set":
                    count = int(cmd.count)
                elif meta[1] in ("del", "done"):
                    del state[cmd.uuid]
    else:
        nodb()

    table = []
    n = 1
    for cmd in state.values():
        table.append((n, cmd))
        n += 1
    return count, table


def now():
    return datetime.now(tz=pytz.UTC)


def serialize(file_, cmd):
    return json.dumps((Meta(now(), cmds_inv[cmd.__class__]), cmd))


def deserialize(ser):
    ret = json.loads(ser)
    meta = ret[0]
    meta = Meta(datetime.fromtimestamp(meta[0], tz=pytz.UTC), meta[1])
    cmd = cmds[meta[1]](*ret[1])
    return meta, cmd


def nodb():
    raise click.ClickException(db_error)
