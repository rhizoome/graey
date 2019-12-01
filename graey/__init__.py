import codecs
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
    estimate = field(initial=0.0)
    duration = field(initial=0.0)

    @property
    def all(self):
        return self.done + self.open

    def __repr__(self):
        return (
            f"est: {self.estimate:.2f}({self.all}) "
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
    with codecs.open("graey.db", "a+") as f:
        s = serialize(f, Add(str(uuid4()), task, " ".join(action), estimate))
        f.write(f"{s}\n")


main.add_command(add)


@click.command(help="show open tasks")
@click.argument("TASK", nargs=-1)
def show(task):
    filter = None
    if task:
        filter = task[0]
    count, table = get_table()
    table = [
        (n, c.task, c.action) for n, c in table if filter is None or c.task == filter
    ]
    if table:
        print(
            tt.to_string(
                table,
                header=["id", "task", "action"],
                style=tt.styles.booktabs,
                alignment="r",
            )
        )
    print(f"  Gräy: {count}")


main.add_command(show)


@click.command("del", help="delete an ACTION by id (see show)")
@click.argument("ACTION", nargs=1, type=click.INT)
def delete(action):
    _, table = get_table()
    cmd = table[action - 1][1]
    with codecs.open("graey.db", "a+") as f:
        s = serialize(f, Del(cmd.uuid))
        f.write(f"{s}\n")


main.add_command(delete)


@click.command(help="complete an ACTION by id (DURATION in HHMM)")
@click.argument("ACTION", nargs=1, type=click.INT)
@click.argument("DURATION", nargs=1, type=click.INT)
def done(action, duration):
    _, table = get_table()
    cmd = table[action - 1][1]
    float_hours = duration_to_hours(duration)
    with codecs.open("graey.db", "a+") as f:
        s = serialize(f, Done(cmd.uuid, float_hours))
        f.write(f"{s}\n")
    print(f"Marked action {action} done in {float_hours} hours")


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
        est = task.estimate
        rem = est - task.duration
        table.append(
            (idx, fmt(task.estimate), fmt(rem), fmt(est * factor), fmt(rem * factor))
        )
    print(
        tt.to_string(
            table,
            alignment="r",
            header=[
                "task",
                "estimate",
                "remaining",
                "estimate (corr)",
                "remaining (corr)",
            ],
            style=tt.styles.booktabs,
        )
    )
    factor = calc_factor(state)
    print(f"Average estimate: {avg_task_estimate(state, factor)}")


main.add_command(tasks)


@click.command("gry", help="set graey COUNT")
@click.argument("COUNT", nargs=1, type=click.INT)
def gry(count):
    with codecs.open("graey.db", "a+") as f:
        s = serialize(f, Gry(count))
        f.write(f"{s}\n")


main.add_command(gry)


@click.command("est", help="set default ESTIMATE")
@click.argument("ESTIMATE", nargs=1, type=click.INT)
def est(estimate):
    estimate = duration_to_hours(estimate)
    with codecs.open("graey.db", "a+") as f:
        s = serialize(f, Est(estimate))
        f.write(f"{s}\n")


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
    part, start, end, grad = predict(calc)
    # pred = end[0] + (end[1] / grad[1] * grad[0])
    pred = 0
    rem = last.estimate - last.duration
    rem_corr = lastc[0] - last.duration
    rem_pred = pred - last.duration
    print(f"Actions:                {last.all:8d}")
    print(f"Actions (done):         {last.done:8d}")
    print(f"Actions (open):         {last.open:8d}")
    print(f"Tasks:                  {tasks:8d}")
    print(f"Tasks (done):           {tasks_done:8d}")
    print(f"Tasks (open):           {tasks_open:8d}")
    print(f"Estimate:                  {last.estimate:8.2f}h")
    print(f"Estimate (corrected):      {lastc[0]:8.2f}h")
    print(f"Estimate (predicted):      {pred:8.2f}h")
    print(f"Prediction data-point:  {part:8d}")
    print(f"Done:                      {last.duration:8.2f}h")
    print(f"Remaining:                 {rem:8.2f}h")
    print(f"Remaining (corrected):     {rem_corr:8.2f}h")
    print(f"Remaining (predicted):     {rem_pred:8.2f}h")


main.add_command(stats)


@click.command(help="display graph")
def graph():
    import_plt()
    matplotlib.use("TkAgg")
    fig, ax = plot()
    try:
        out = check_output(["xrdb", "-query"])
        for line in out.decode("UTF-8").splitlines():
            if line.startswith("Xft.dpi"):
                splt = [x for x in line.split() if x]
                fig.set_dpi(int(splt[1]))
    except CalledProcessError:
        pass
    plt.show()


main.add_command(graph)


@click.command(help="save graph")
def save():
    import_plt()
    fig, ax = plot()
    mpld3.save_html(fig, "graey.html")


main.add_command(save)


@click.command(help="output as csv (estimate, done)")
def csv():
    for state in get_states():
        est, done = calculate(state)
        print(est, done)


main.add_command(csv)


def fmt(value):
    return f"{value:8.2f}"


def get_default_est():
    estimate = 1.0
    if os.path.exists("graey.db"):
        with codecs.open("graey.db", "r") as f:
            for line in f:
                meta, cmd = deserialize(line)
                if meta.cmd == "est":
                    estimate = cmd.estimate
    return estimate


def duration_to_hours(duration):
    hours = duration // 100
    minutes = duration - (hours * 100)
    if minutes >= 60:
        raise click.ClickException(minute_error)
    return (60 * hours + minutes) / 60


def predict(calc):
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
    ax.set_xlabel("estimated effort (hours)")
    ax.set_ylabel("open effort (hours)")


def plot_velocity(ax, calc):
    part, start, end, grad = predict(calc)
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


def plot():
    states = list(get_states())
    calc = [calculate(state) for state in states]
    fig, ax = plt.subplots()
    plot_effort(ax, calc)
    part = plot_velocity(ax, calc)
    ax.legend(["data", f"trend ({part} data-points)"])
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


def avg_task_estimate(state, factor):
    tasks = state.tasks
    if tasks:
        estimate = 0
        for task in tasks.values():
            all = task.all
            unknown = 0
            if all < 4 and task.open:
                unknown = 4 - all
            estimate += (
                unknown * state.default_est * factor
                + (task.estimate - task.duration) * factor
                + task.duration
            )

        return estimate / len(tasks)
    else:
        return state.default_est * 4


def calc_factor(state):
    if state.duration and state.done and state.open:
        return (state.duration / state.done) / (state.estimate / state.all)
    else:
        return 1


def calculate(state):
    factor = calc_factor(state)
    avg_estimate = avg_task_estimate(state, factor)
    estimate = state.estimate * factor + state.graey * avg_estimate
    done = state.duration
    assert estimate >= done
    return estimate, done


def update_state(state, line):
    actions = state.actions
    tasks = state.tasks
    meta, cmd = line
    if meta.cmd == "add":
        actions = actions.set(cmd.uuid, cmd)
        task = tasks.get(cmd.task, default_task)
        task = task.set(open=task.open + 1, estimate=task.estimate + cmd.estimate)
        tasks = tasks.set(cmd.task, task)
        return state.set(
            actions=actions,
            tasks=tasks,
            open=state.open + 1,
            estimate=state.estimate + cmd.estimate,
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
        task = task.set(open=task.open - 1, estimate=task.estimate - action.estimate)
        tasks = tasks.set(key, task)
        return state.set(
            actions=actions,
            tasks=tasks,
            open=state.open - 1,
            estimate=state.estimate - action.estimate,
        )
    if meta.cmd == "done":
        actions = actions.remove(cmd.uuid)
        task = task.set(
            open=task.open - 1,
            done=task.done + 1,
            estimate=task.estimate + cmd.duration - action.estimate,
            duration=task.duration + cmd.duration,
        )
        tasks = tasks.set(key, task)
        return state.set(
            open=state.open - 1,
            done=state.done + 1,
            actions=actions,
            tasks=tasks,
            estimate=state.estimate + cmd.duration - action.estimate,
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


db_error = "graey.db does not exist"
minute_error = """A duration has the format HHMM, the minute part must always be lower than 60

Correct: 20, 100, 0120, 120, 230, 0359
Incorrect: 70, 170, 199"""
