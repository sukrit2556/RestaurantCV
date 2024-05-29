from multiprocessing import cpu_count

print(cpu_count())
if cpu_count() > 2:
        worker_num = cpu_count() - 1  # 1 for capturing frames

print(worker_num)