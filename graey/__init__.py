import codecs
from collections import OrderedDict, namedtuple
from datetime import datetime
from subprocess import CalledProcessError, check_output
from uuid import uuid4

import click
import matplotlib
import matplotlib.pyplot as plt
import mpld3
import numpy as np
import pytz
import termtables as tt
from pyrsistent import PRecord, field, pmap

try:
    import ujson as json
except ModuleNotFoundError:
    import json


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
                table, header=["id", "task", "action"], style=tt.styles.booktabs
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
    table = []
    tasks = state.tasks
    for task in tasks.keys():
        table.append((task, tasks[task].duration))
    print(tt.to_string(table, header=["task", "duration"], style=tt.styles.booktabs))
    print(f"Average: {avg_task_duration(state)} (cannot be smaller than 4)")


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
    pass


main.add_command(stats)


@click.command(help="display graph")
def graph():
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
    fig, ax = plot()
    mpld3.save_html(fig, "graey.html")


main.add_command(save)


@click.command(help="output as csv (estimate, done)")
def csv():
    for state in get_states():
        est, done = calculate(state)
        print(est, done)


main.add_command(csv)


def get_default_est():
    estimate = 1.0
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


def plot_effort(ax, calc):
    xs = []
    ys = []
    for x, y in calc:
        xs.append(x)
        ys.append(x - y)
    ax.plot(xs, ys)
    ax.set_xlabel("estimated effort (hours)")
    ax.set_ylabel("open effort (hours)")


def plot_velocity(ax, calc):
    len_calc = len(calc)
    part = len_calc // 3
    start_est = calc[-part][0]
    start_dur = calc[-part][1]
    grad_est = (calc[-1][0] - start_est) / part
    grad_dur = (calc[-1][1] - start_dur) / part
    xs = np.linspace(start_est, start_est + grad_est * len_calc, len_calc)
    ys = np.linspace(start_dur, start_dur + grad_dur * len_calc, len_calc)
    ds = xs - ys
    zero = np.where(np.diff(np.sign(ds)))[0][0]
    ax.plot(xs[:zero], ds[:zero])
    return zero


def plot():
    fig, ax = plt.subplots()
    states = list(get_states())
    calc = [calculate(state) for state in states]
    plot_effort(ax, calc)
    data_points = plot_velocity(ax, calc)
    ax.legend(["data", f"trend ({data_points} data-points)"])
    plt.title("Effort proportion")
    return fig, ax


def get_states():
    state = State()
    with codecs.open("graey.db", "r") as f:
        for line in f:
            line = deserialize(line)
            state = update_state(state, line)
            yield state


def avg_task_duration(state, factor):
    tasks = state.tasks
    if tasks:
        duration = 0
        for task in tasks.values():
            open = task.open
            done = task.done
            all = task.all
            if all < 4:
                open += 4 - all
            assert open + done >= 4
            duration += open * state.default_est * factor + task.duration
        return duration / len(tasks)
    else:
        return state.default_est * 4


def calculate(state):
    if state.duration and state.done and state.open:
        factor = (state.duration / state.done) / (state.estimate / state.all)
    else:
        factor = 1
    avg_duration = avg_task_duration(state, factor)
    estimate = state.estimate * factor + state.graey * avg_duration
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
    with codecs.open("graey.db", "r") as f:
        for line in f:
            meta, cmd = deserialize(line)
            if meta[1] == "add":
                state[cmd.uuid] = cmd
            elif meta[1] == "set":
                count = int(cmd.count)
            else:
                del state[cmd.uuid]

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


minute_error = """A duration has the format HHMM, the minute part must always be lower than 60

Correct: 20, 100, 0120, 120, 230, 0359
Incorrect: 70, 170, 199"""
