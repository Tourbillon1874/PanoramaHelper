import sys
import os
import importlib
import json
import urllib.request
import zipfile
import io
import shutil


CURRENT_VERSION = "1.0.2"

PACKAGE_NAME = __name__
PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))

class _NullWriter:
    def write(self, s): pass
    def flush(self): pass

_original_stdout = sys.stdout
_original_stderr = sys.stderr

def _silent():
    sys.stdout = _NullWriter()
    sys.stderr = _NullWriter()

def _restore():
    sys.stdout = _original_stdout
    sys.stderr = _original_stderr

def _check_and_update():
    try:
        api_url = "https://api.github.com/repos/Tourbillon1874/PanoramaHelper/releases/latest"
        req = urllib.request.Request(api_url, headers={"User-Agent": "PanoramaHelper-Updater"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
        latest = data["tag_name"].lstrip("v")
    except Exception:
        return False

    if latest == CURRENT_VERSION:
        return False

    try:
        zip_url = f"https://github.com/Tourbillon1874/PanoramaHelper/archive/refs/tags/v{latest}.zip"
        with urllib.request.urlopen(zip_url, timeout=15) as resp:
            zip_data = resp.read()
    except Exception:
        return False

    try:
        with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
            root_folder = zf.namelist()[0].split('/')[0]
            tmp_dir = os.path.join(PLUGIN_DIR, "_tmp_update")
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)
            zf.extractall(tmp_dir)
            src = os.path.join(tmp_dir, root_folder)

        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(PLUGIN_DIR, item)
            if os.path.isdir(s):
                if os.path.exists(d):
                    shutil.rmtree(d, ignore_errors=True)
                shutil.copytree(s, d)
            else:
                if os.path.exists(d):
                    tmp_file = d + ".tmp"
                    shutil.copy2(s, tmp_file)
                    os.replace(tmp_file, d)
                else:
                    shutil.copy2(s, d)

        shutil.rmtree(tmp_dir, ignore_errors=True)
        return True
    except Exception:
        return False

_silent()
did_update = _check_and_update()
_restore()

if did_update:
    prefix = PACKAGE_NAME + "."
    for mod in list(sys.modules.keys()):
        if mod.startswith(prefix) and mod != PACKAGE_NAME:
            del sys.modules[mod]

try:
    _core = importlib.import_module("._core", package=PACKAGE_NAME)
except Exception:
    class _Fallback:
        pass
    _core = _Fallback()

NODE_CLASS_MAPPINGS = getattr(_core, "NODE_CLASS_MAPPINGS", {})
NODE_DISPLAY_NAME_MAPPINGS = getattr(_core, "NODE_DISPLAY_NAME_MAPPINGS", {})
