import os
import signal
import subprocess
import sys
import tempfile


def run(args: list[str], timeout: int, env: dict[str, str] = {}) -> (str, str):
    args = list(map(str, args))
    env = {**os.environ, **env}
    kwargs = {"env": env}
    tmp_stdout, tmp_stderr = tempfile.TemporaryFile(), tempfile.TemporaryFile()
    kwargs.update({"stdout": tmp_stdout, "stderr": tmp_stderr})

    stdout = ""
    stderr = ""
    try:
        proc = subprocess.Popen(args, **kwargs)
        proc_timeout = False
        try:
            proc.wait(timeout)
        except subprocess.TimeoutExpired as e:
            proc.kill()
            proc_timeout = True
        success = not proc_timeout and proc.returncode == 0

        stdout = _clean_output_file(tmp_stdout)
        stderr = _clean_output_file(tmp_stderr)

        if not success:
            if not proc_timeout:
                msg = "Program failed with return code={}:".format(proc.returncode)
            else:
                msg = "Program timeout"
            raise ExternalProgramRunError(msg, args, env=env, stdout=stdout, stderr=stderr)
    finally:
        tmp_stderr.close()
        tmp_stdout.close()
    return (stdout, stderr)


def _clean_output_file(file_object):
    file_object.seek(0)
    return "".join(map(lambda s: s.decode("utf-8"), file_object.readlines()))


class ExternalProgramRunContext:
    def __init__(self, proc):
        self.proc = proc

    def __enter__(self):
        self.__old_signal = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, self.kill_job)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is KeyboardInterrupt:
            self.kill_job()
        signal.signal(signal.SIGTERM, self.__old_signal)

    def kill_job(self, captured_signal=None, stack_frame=None):
        self.proc.kill()
        if captured_signal is not None:
            # adding 128 gives the exit code corresponding to a signal
            sys.exit(128 + captured_signal)


class ExternalProgramRunError(RuntimeError):
    def __init__(self, message, args, env=None, stdout=None, stderr=None):
        super(ExternalProgramRunError, self).__init__(message, args, env, stdout, stderr)
        self.message = message
        self.args = args
        self.env = env
        self.out = stdout
        self.err = stderr

    def __str__(self):
        info = self.message
        info += "\nCOMMAND: {}".format(" ".join(self.args))
        info += "\nSTDOUT: {}".format(self.out or "[empty]")
        info += "\nSTDERR: {}".format(self.err or "[empty]")
        return info
