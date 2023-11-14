from flexcv.run import Run


def test_run_methods_no_exception():
    # Test all methods
    run = Run()
    run.fetch()
    run.stop()
    temp = run["key"]
    run["key"] = "value"
    getattr(run, "key")
    setattr(run, "key", "value")
    delattr(run, "key")
    str(run)
    repr(run)
    run.append("value")
    run.log("message")
    run.upload("file_path")
