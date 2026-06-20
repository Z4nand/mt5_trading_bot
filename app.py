import importlib
import sys


MODULE_NAME = "src.ui.streamlit_app"

if MODULE_NAME in sys.modules:
    importlib.reload(sys.modules[MODULE_NAME])
else:
    importlib.import_module(MODULE_NAME)
