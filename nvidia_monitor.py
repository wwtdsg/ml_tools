import os
import re
import time
import matplotlib
import matplotlib.pyplot as plt

def get_mem(pid, gpu):
    cmd_out = os.popen("nvidia-smi").read()
    pid_line = [line for line in cmd_out.split('\n') if re.search("{} +{}".format(gpu, pid), line)]
    if len(pid_line) > 1:
        print("????")
    elif not pid_line:
        return -1
    mem = re.search("\d+MiB", pid_line[0])
    return int(pid_line[0][mem.span()[0]: mem.span()[1] - 3])


def mem_monitor(pid, gpu, max_t):
    interval = 0.2
    plt.figure(1)
    x = 0
    max_iter = max_t * 5
    while max_iter > 0:
        mem = get_mem(pid, gpu)
        if mem < 0:
            break
        plt.plot(x, mem, color="blue", linewidth=2.5, linestyle="-")
        x = x + interval
        time.sleep(interval)
        max_iter -= 1
    plt.savefig('/home/chen/wt/tools/mem.png')

if __name__ == '__main__':
    mem_monitor(14444, 2, 5)
