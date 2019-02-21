import django
import cloudpickle
import multiprocessing as mp
import os

from rekall.runtime import (
        AbstractWorkerPool,
        Runtime,
        SpawnedProcessPool,
        get_forked_process_pool_factory,
        get_spawned_process_pool_factory)

def get_worker_pool_factory_for_jupyter(num_workers=mp.cpu_count()):
    fork_factory = get_forked_process_pool_factory(num_workers)
    # Forked Django Database connections will not work. Hence we close all
    # connections before forking to allow child processes to establish their
    # own connections.
    def worker_pool_factory(fn):
        django.db.connections.close_all()
        return fork_factory(fn)
    return worker_pool_factory

# Jupyter Notebook needs to use forking so that child processes can correctly
# connect to Jupyter.
def get_runtime_for_jupyter(num_workers=mp.cpu_count()):
    return Runtime(get_worker_pool_factory_for_jupyter(num_workers))

def _set_up_django():
    # set up django context
    import django
    django.setup()

def get_worker_pool_factory_for_script(num_workers=mp.cpu_count()):
    def worker_pool_factory(fn):
        return SpawnedProcessPool(fn, num_workers, initializer=_set_up_django)
    return worker_pool_factory

# Spawning is preferred because it plays well with multithreading libraries,
# but it does not work in Jupyter Notebook.
def get_runtime_for_script(num_workers=mp.cpu_count()):
    return Runtime(get_worker_pool_factory_for_script(num_workers))

# A Worker Pool that writes to a shared storage instead of sending results to
# master process in memory
class _WithStorage(AbstractWorkerPool):
    class Result():
        def __init__(self, filename_future):
            self._filename_future = filename_future
        def get(self):
            filename = self._filename_future.get()
            with open(filename, 'rb') as f:
                return cloudpickle.load(f)

    def __init__(self, fn, pool_factory, output_dir):
        os.makedirs(output_dir)
        self._pool = pool_factory(_WithStorage.wrap(fn, output_dir))

    # Wraps the query function to write to output_dir
    @staticmethod
    def wrap(query, output_dir):
        def fn(vids):
            results = query(vids)
            filename = os.path.join(
                    output_dir,
                    "{0}({1}).pickle".format(vids[0], len(vids)))
            with open(filename, 'wb') as f:
                cloudpickle.dump(results, f)
            return filename
        return fn

    def apply_async(self, vids, callback):
        filename_future = self._pool.apply_async(vids, callback)
        return _WithStorage.Result(filename_future)
    
    def shut_down(self):
        return self._pool.shut_down()

# A WorkerPoolFactory that writes results to storage
class WorkerPoolWithStorageFactory():
    def __init__(self, output_dir, factory):
        self._call_index = 0
        self._output_dir = output_dir
        self._factory = factory

    def get_output_dir(self):
        return os.path.join(self._output_dir, str(self._call_index))

    def __call__(self, fn):
        output_dir = self.get_output_dir()
        self._call_index += 1
        return _WithStorage(fn, self._factory, output_dir)


