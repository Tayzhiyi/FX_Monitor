from dash import Dash

from src.dashboard_data import (
    PORTFOLIO,
)

from src.dashboard_ui import (
    build_layout,
)

from src.dashboard_callbacks import (
    register_callbacks,
)


# ============================================================
# DASH APPLICATION
# ============================================================

app = Dash(
    __name__
)

server = app.server

app.title = (
    "FX Portfolio Dashboard"
)


# ============================================================
# LAYOUT
# ============================================================

app.layout = (
    build_layout(
        PORTFOLIO
    )
)


# ============================================================
# CALLBACKS
# ============================================================

register_callbacks(
    app
)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True,
    )