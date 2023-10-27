from neptune.metadata_containers.run import Run as NeptuneRun


class Run(NeptuneRun):
    """Dummy Run class inheriting from Neptune Run.
    It can be used to pass to all functions and methods that expect a Neptune Run Object.
    It will not do anything."""
    def __init__(self, *args, **kwargs):
        self.run_id = "dummy"

    def fetch(self):
        return self.run_id

    def stop(self):
        pass

    def __getitem__(self, key):
        return self

    def __setitem__(self, key, value):
        pass

    def __getattr__(self, key):
        return self

    def __setattr__(self, key, value):
        pass

    def __delattr__(self, key):
        pass

    def __str__(self):
        return self.run_id

    def __repr__(self):
        return self.run_id

    def append(self, *args, **kwargs):
        pass

    def log(self, *args, **kwargs):
        pass

    def upload(self, *args, **kwargs):
        pass


if __name__ == "__main__":
    pass
