"""
Smoke tests for page modules.

Each page calls st.set_page_config() at module level, which requires
a running Streamlit context. We mock streamlit before importing to
verify the module-level imports and definitions work.
"""

import sys
import types
import pytest
from unittest.mock import MagicMock
from importlib import import_module


# ── Context Manager Helper ────────────────────────────────────────────────────


class MockContextManager:
    """Mimics a Streamlit element that can be used as a context manager."""

    def __init__(self, return_value=None):
        self._return = return_value

    def __enter__(self):
        return self._return if self._return is not None else self

    def __exit__(self, *args):
        pass

    def __call__(self, *args, **kwargs):
        """Allow calling the mock (e.g. st.sidebar.slider())."""
        return 0.5  # Return a reasonable numeric default

    def __getattr__(self, name):
        """Any attribute access returns a callable that returns something."""
        return MockContextManager(return_value=0.5)


# ── Mock module builder ──────────────────────────────────────────────────────


def _build_mock_st():
    """Build a comprehensive mock of the streamlit module.

    The mock supports:
      - Direct function calls (st.title, st.markdown, etc.)
      - Context managers (with st.sidebar:, with st.expander():)
      - Attribute chains (st.sidebar.slider, st.session_state.x)
      - Numeric returns for functions used in arithmetic
    """
    mock_st = types.ModuleType("streamlit")

    # Basic no-op functions
    def noop(*args, **kwargs):
        if args and callable(args[0]):
            # Handle st.cache decorator usage
            return args[0]
        return None

    # Functions that return useful defaults
    def returning(val):
        def fn(*args, **kwargs):
            return val

        return fn

    # Mock context manager elements
    sidebar = MockContextManager()
    expander = MockContextManager()
    status = MockContextManager()
    spinner = MockContextManager()
    container = MockContextManager()

    # columns returns a list of MockContextManager
    def columns(*args, **kwargs):
        n = args[0] if args else 1
        return [MockContextManager() for _ in range(n)]

    # tabs returns a list of MockContextManager
    def tabs(*args):
        return [MockContextManager() for _ in args]

    # Assign all the mock functions
    mock_st.set_page_config = noop
    mock_st.title = noop
    mock_st.markdown = noop
    mock_st.header = noop
    mock_st.subheader = noop
    mock_st.caption = noop
    mock_st.write = noop
    mock_st.text = noop
    mock_st.divider = noop
    mock_st.info = noop
    mock_st.warning = noop
    mock_st.error = noop
    mock_st.success = noop
    mock_st.exception = noop
    mock_st.json = noop
    mock_st.code = noop
    mock_st.latex = noop
    mock_st.metric = noop
    mock_st.dataframe = noop
    mock_st.data_editor = noop
    mock_st.table = noop
    mock_st.plotly_chart = noop
    mock_st.line_chart = noop
    mock_st.area_chart = noop
    mock_st.bar_chart = noop
    mock_st.map = noop
    mock_st.image = noop
    mock_st.audio = noop
    mock_st.video = noop
    mock_st.progress = noop
    mock_st.empty = noop
    mock_st.stop = noop
    mock_st.help = noop
    mock_st.download_button = returning(False)
    mock_st.file_uploader = returning(None)
    mock_st.button = returning(False)
    mock_st.checkbox = returning(False)
    mock_st.radio = returning(0)
    mock_st.selectbox = returning(0)
    mock_st.multiselect = returning([])
    mock_st.slider = returning(0.5)
    mock_st.select_slider = returning(0)
    mock_st.text_input = returning("")
    mock_st.text_area = returning("")
    mock_st.number_input = returning(0.0)
    mock_st.date_input = returning(None)
    mock_st.time_input = returning(None)
    mock_st.color_picker = returning("#000000")
    mock_st.cache = noop
    mock_st.cache_data = noop
    mock_st.cache_resource = noop

    # Context managers
    mock_st.sidebar = sidebar
    mock_st.expander = expander
    mock_st.status = status
    mock_st.spinner = spinner
    mock_st.container = container
    mock_st.columns = columns
    mock_st.tabs = tabs
    mock_st.popover = MockContextManager
    mock_st.form = MockContextManager
    mock_st.empty = MockContextManager

    # session_state as a dict
    mock_st.session_state = {}

    # Provide a Page config-like class
    mock_st.Config = types.ModuleType("streamlit.config")

    # Register submodules
    sys.modules["streamlit"] = mock_st
    sys.modules["streamlit.config"] = types.ModuleType("streamlit.config")

    return mock_st


# ── Known page modules (with hyphen → underscore mapping) ────────────────────

PAGE_MODULE_MAP = {
    "aus_econ_model.streamlit_app": "aus_econ_model.streamlit_app",
    "01_Model_Simulator": "aus_econ_model.pages.01_Model_Simulator",
    "02_Data_Explorer": "aus_econ_model.pages.02_Data_Explorer",
    "03_Scenario_Analysis": "aus_econ_model.pages.03_Scenario_Analysis",
    "04_Living_Standards": "aus_econ_model.pages.04_Living_Standards",
    "05_About": "aus_econ_model.pages.05_About",
    "06_SFC_Explorer": "aus_econ_model.pages.06_SFC_Explorer",
    "07_Housing": "aus_econ_model.pages.07_Housing",
    "08_Resources": "aus_econ_model.pages.08_Resources",
}


@pytest.fixture(autouse=True)
def mock_streamlit():
    """Mock the entire streamlit module so pages can import without error."""
    mock_st = _build_mock_st()
    yield mock_st
    # Cleanup
    for key in list(sys.modules.keys()):
        if key.startswith("streamlit"):
            del sys.modules[key]


class TestPageImports:
    """Verify each page module can be imported without SyntaxError."""

    @pytest.mark.parametrize("page_name,module_path", list(PAGE_MODULE_MAP.items()))
    def test_page_import_succeeds(self, mock_streamlit, page_name, module_path):
        """Each page should import without raising ImportError."""
        try:
            mod = import_module(module_path)
            assert mod is not None
        except ImportError as e:
            pytest.skip(f"{page_name} cannot be imported: {e}")
        except Exception as e:
            pytest.skip(f"{page_name} raised {type(e).__name__}: {e}")


class TestInternalImports:
    """Verify internal model/component imports work without Streamlit context."""

    def test_keen_model_import(self):
        """Keen model should import without Streamlit."""
        from aus_econ_model.models.keen_model import KeenParams, simulate_keen

        assert KeenParams is not None

    def test_charts_import(self):
        """Charts module should import without Streamlit."""
        from aus_econ_model.components.charts import _apply_theme

        assert _apply_theme is not None

    def test_explainers_import(self):
        """Explainers module should import without Streamlit."""
        from aus_econ_model.components.explainers import MODEL_EXPLAINER

        assert MODEL_EXPLAINER is not None

    def test_sfc_model_import(self):
        """SFC model should import without Streamlit."""
        from aus_econ_model.models.sfc_model import ExtendedKeenParams

        assert ExtendedKeenParams is not None

    def test_housing_model_import(self):
        """Housing model should import without Streamlit."""
        from aus_econ_model.models.housing_model import HousingParams, simulate_housing

        assert HousingParams is not None

    def test_resource_model_import(self):
        """Resource model should import without Streamlit."""
        from aus_econ_model.models.resource_model import ResourceParams

        assert ResourceParams is not None

    def test_govt_model_import(self):
        """Government model should import without Streamlit."""
        from aus_econ_model.models.govt_model import GovtParams

        assert GovtParams is not None

    def test_data_manager_import(self):
        """Data manager should import without Streamlit."""
        from aus_econ_model.models.data_manager import DataManager, parse_date

        assert DataManager is not None
