"""Amostragem de memória da execução e dos subprocessos no macOS."""

import ctypes
import os
import sys
import threading
import time


class Recursos:
    def __init__(self):
        self.inicio = time.monotonic()
        self.parar = threading.Event()
        self.picos = {}
        self.thread = None
        if sys.platform == "darwin":
            self.lib = ctypes.CDLL("/usr/lib/libproc.dylib")
            self.thread = threading.Thread(target=self._acompanhar, daemon=True)
            self.thread.start()

    def _acompanhar(self):
        while True:
            pendentes, vistos = [os.getpid()], set()
            rss = fisica = 0
            while pendentes:
                pid = pendentes.pop()
                if pid in vistos:
                    continue
                vistos.add(pid)
                dados = (ctypes.c_uint64 * 32)()
                if self.lib.proc_pid_rusage(pid, 2, ctypes.byref(dados)) == 0:
                    rss += dados[8]
                    fisica += dados[9]
                filhos = (ctypes.c_int * 512)()
                n = self.lib.proc_listchildpids(pid, filhos, ctypes.sizeof(filhos))
                pendentes.extend(filhos[:max(0, n)])
            for k, v in (("rss_arvore_bytes", rss), ("memoria_fisica_arvore_bytes", fisica),
                         ("processos_observados", len(vistos))):
                self.picos[k] = max(self.picos.get(k, 0), v)
            if self.parar.wait(1):
                break

    def fechar(self):
        self.parar.set()
        if self.thread:
            self.thread.join()
        return {"segundos": time.monotonic()-self.inicio, "picos_amostrados_1s": self.picos}
