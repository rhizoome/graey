import codecs
from collections import OrderedDict, defaultdict, namedtuple
from datetime import datetime
from uuid import uuid4

import click
import mpld3
import pytz
import termtables as tt
from pyrsistent import PRecord, field, pmap

try:
    import ujson as json
except ModuleNotFoundError:
    import json

action_duration = 1.0


def invert(dict_):
    ret = {}
    for key in dict_.keys():
        value = dict_[key]
        ret[value] = key
    return ret


Meta = namedtuple("Meta", ("timestamp", "cmd"))
Add = namedtuple("Add", ("uuid", "task", "action"))
Del = namedtuple("Del", ("uuid"))
Done = namedtuple("Done", ("uuid", "duration"))
Set = namedtuple("Set", ("count"))
cmds = {"add": Add, "del": Del, "done": Done, "set": Set}
cmds_inv = invert(cmds)


class Task(PRecord):
    count = field(initial=0)
    duration = field(initial=0)


default_task = Task()


class State(PRecord):
    open = field(initial=0)
    done = field(initial=0)
    duration = field(initial=0.0)
    gräy = field(initial=0)
    actions = field(initial=pmap())
    tasks = field(initial=pmap())


@click.group()
def main():
    pass


@click.command(help="add ACTION to TASK")
@click.argument("TASK")
@click.argument("ACTION", nargs=-1)
def add(task, action):
    with codecs.open("gräy.db", "a+") as f:
        s = serialize(f, Add(str(uuid4()), task, " ".join(action)))
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
    with codecs.open("gräy.db", "a+") as f:
        s = serialize(f, Del(cmd.uuid))
        f.write(f"{s}\n")


main.add_command(delete)


@click.command(help="complete an ACTION by id (DURATION in HHMM)")
@click.argument("ACTION", nargs=1, type=click.INT)
@click.argument("DURATION", nargs=1, type=click.INT)
def done(action, duration):
    _, table = get_table()
    cmd = table[action - 1][1]
    hours = duration // 100
    minutes = duration - (hours * 100)
    assert minutes < 60
    float_hours = (60 * hours + minutes) / 60
    with codecs.open("gräy.db", "a+") as f:
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


@click.command("set", help="set gräy COUNT")
@click.argument("COUNT", nargs=1, type=click.INT)
def set_(count):
    with codecs.open("gräy.db", "a+") as f:
        s = serialize(f, Set(count))
        f.write(f"{s}\n")


main.add_command(set_)


@click.command(help="display stats")
def stats():
    pass


main.add_command(stats)


@click.command(help="display graph")
def graph():
    for state in get_states():
        print(calculate(state))


main.add_command(graph)


@click.command(help="save graph")
def save():
    pass


main.add_command(save)


def get_states():
    state = State()
    with codecs.open("gräy.db", "r") as f:
        for line in f:
            line = deserialize(line)
            state = update_state(state, line)
            yield state


def avg_task_duration(state):
    tasks = state.tasks
    if tasks:
        avg_duration = sum((task.duration for task in tasks.values())) / len(tasks)
        if avg_duration < 4:
            avg_duration = 4
    else:
        avg_duration = 4
    return avg_duration


def calculate(state):
    avg_duration = avg_task_duration(state)
    if state.done:
        factor = state.duration / state.done
    else:
        factor = 1
    estimate = state.open * factor + state.duration + state.gräy * avg_duration
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
        task = task.set(count=task.count + 1)
        tasks = tasks.set(cmd.task, task)
        state = state.set(actions=actions, tasks=tasks)
        return state.set(open=state.open + 1)
    if meta.cmd == "set":
        return state.set(gräy=cmd.count)
    key = actions[cmd.uuid].task
    task = tasks[key]
    if meta.cmd == "del":
        tasks = tasks.set(key, task.set(count=task.count - 1))
        actions = actions.remove(cmd.uuid)
        state = state.set(actions=actions, tasks=tasks)
        return state.set(open=state.open - 1)
    if meta.cmd == "done":
        tasks = tasks.set(key, task.set(duration=task.duration + cmd.duration))
        actions = actions.remove(cmd.uuid)
        state = state.set(actions=actions, tasks=tasks)
        state = state.set(duration=state.duration + cmd.duration)
        return state.set(open=state.open - 1, done=state.done + 1)
    assert False


def get_table():
    state = OrderedDict()
    count = 0
    with codecs.open("gräy.db", "r") as f:
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
