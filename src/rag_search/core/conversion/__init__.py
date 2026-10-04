"""Document conversion tracking and routing (see docs/design/document-conversion-plan.md).

Everything in this package that the dashboard, the API and the CLI import (``trace``, ``costs``,
``router``, ``profiler``, ``runview``, ``estimate``) is stdlib-only at import time; the PDF/image
libraries are imported lazily, inside the functions that need them, so the light layer never pulls
in numpy, torch or docling.
"""
