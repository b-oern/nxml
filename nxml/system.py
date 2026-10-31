import os
import platform

from nwebclient import base as b
from nwebclient import web as w


def format_bytes(n):
    if n is None:
        return '-'
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(n) < 1024 or unit == 'TB':
            return f"{n:.1f} {unit}" if unit != 'B' else f"{n} B"
        n /= 1024


def select_device(device_mode='auto'):
    """
        :param device_mode: "cpu", "gpu" (CUDA, sonst MPS) oder "auto" (GPU falls vorhanden, sonst CPU)
        :return: torch-Device "cpu", "cuda" oder "mps"
    """
    import torch
    if device_mode not in ('cpu', 'gpu', 'auto'):
        raise ValueError(f'device_mode must be "cpu", "gpu" or "auto", got {device_mode!r}')
    if device_mode == 'cpu':
        return 'cpu'
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    if device_mode == 'gpu':
        raise RuntimeError('device_mode="gpu", but no CUDA or MPS device is available')
    return 'cpu'


class AISystemInfo:
    """
    Mixin für Runner, die KI-Modelle laden: zeigt RAM, CPU, GPU und GPU-RAM.

    class MyRunner(AISystemInfo, r.BaseJobExecutor): ...
    Link: self.link(self.part_systeminfo)
    """

    def system_info(self):
        info = {
            'platform': platform.platform(),
            'python': platform.python_version(),
            'cpu': platform.processor() or platform.machine(),
            'cpu_count': os.cpu_count(),
        }
        try:
            import psutil
            mem = psutil.virtual_memory()
            info['ram_total'] = mem.total
            info['ram_available'] = mem.available
            info['ram_percent'] = mem.percent
            info['process_ram'] = psutil.Process().memory_info().rss
        except ImportError:
            info['ram_total'] = None
        info['gpus'] = []
        try:
            import torch
            info['torch'] = torch.__version__
            info['cuda_available'] = torch.cuda.is_available()
            info['cuda_version'] = torch.version.cuda
            info['mps_available'] = torch.backends.mps.is_available()
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    free, total = torch.cuda.mem_get_info(i)
                    props = torch.cuda.get_device_properties(i)
                    info['gpus'].append({
                        'index': i,
                        'name': props.name,
                        'capability': f"{props.major}.{props.minor}",
                        'memory_total': total,
                        'memory_free': free,
                        'memory_used': total - free,
                        'process_allocated': torch.cuda.memory_allocated(i),
                        'process_reserved': torch.cuda.memory_reserved(i),
                    })
        except ImportError:
            info['torch'] = None
        return info

    def part_systeminfo(self, p: b.Page, params={}):
        p.h1("Systeminfo")
        p.div(w.a("Zurück", self.link(self.part_index)))
        info = self.system_info()
        with p.section(h="System"):
            p.prop("Plattform", info['platform'])
            p.prop("Python", info['python'])
            p.prop("CPU", f"{info['cpu']} ({info['cpu_count']} Threads)")
        with p.section(h="RAM"):
            if info.get('ram_total') is None:
                p.div("psutil nicht installiert")
            else:
                p.prop("Gesamt", format_bytes(info['ram_total']))
                p.prop("Verfügbar", format_bytes(info['ram_available']))
                p.prop("Belegt", f"{info['ram_percent']} %")
                p.prop("Dieser Prozess", format_bytes(info['process_ram']))
        with p.section(h="GPU"):
            if info.get('torch') is None:
                p.div("torch nicht installiert")
            else:
                p.prop("torch", info['torch'])
                p.prop("CUDA", f"{info['cuda_available']} ({info['cuda_version'] or '-'})")
                p.prop("MPS", info['mps_available'])
                for gpu in info['gpus']:
                    p.h3(f"GPU {gpu['index']}: {gpu['name']}")
                    p.prop("Compute Capability", gpu['capability'])
                    p.prop("GPU-RAM gesamt", format_bytes(gpu['memory_total']))
                    p.prop("GPU-RAM frei", format_bytes(gpu['memory_free']))
                    p.prop("GPU-RAM belegt", format_bytes(gpu['memory_used']))
                    p.prop("Dieser Prozess (allocated / reserved)",
                           format_bytes(gpu['process_allocated']) + ' / ' + format_bytes(gpu['process_reserved']))
