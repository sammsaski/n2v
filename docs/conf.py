# Configuration file for the Sphinx documentation builder.

import os
import sys

# Add project root to path so autodoc can import n2v
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "n2v"
copyright = "2025, VeriVITAL Team"
author = "VeriVITAL Team"

try:
    import n2v

    version = n2v.__version__
    release = n2v.__version__
except ImportError:
    version = "0.1.0"
    release = "0.1.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    # Exclude original markdown files that have been copied into subdirectories
    "development_status.md",
    "lp_solvers.md",
    "probabilistic_verification.md",
    "theoretical_foundations.md",
]

# -- Autodoc configuration --------------------------------------------------

autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
# Prevent autodoc-generated members from appearing in the right-hand "On this page" ToC
toc_object_entries = False
autodoc_mock_imports = [
    "onnx2torch",
    "onnx",
    "onnxruntime",
    "cvxpy",
    "torch",
    "numpy",
    "scipy",
    "torchvision",
]

# -- Napoleon configuration -------------------------------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True

# -- MyST configuration -----------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
]
myst_heading_anchors = 4
suppress_warnings = ["myst.xref_missing"]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- Intersphinx configuration ----------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

# -- HTML output configuration ----------------------------------------------

html_theme = "furo"
html_title = "n2v"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]

html_theme_options = {
    "source_repository": "https://github.com/sammsaski/n2v",
    "source_branch": "main",
    "source_directory": "docs/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/sammsaski/n2v",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
}
