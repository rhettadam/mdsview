"""Graphical interface for browsing and analysing MITgcm output."""

from __future__ import annotations

import contextlib
import os

# Must run before pyplot import so matplotlib does not load Qt on Windows.
os.environ.setdefault("MPLBACKEND", "TkAgg")
import matplotlib

matplotlib.use("TkAgg")

import threading
import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from . import animation, catalog, grid, gui_theme, io, ops, plotting
from .colormaps import COLORMAP_CHOICES, DEFAULT_DIFF_CMAP, DEFAULT_PLOT_CMAP

C = gui_theme.COLORS
G = gui_theme


class _IndexSlider:
    """Slider, entry, and step buttons for iteration numbers or level indices."""

    _DEFAULT_MAX_SLIDER_STEPS = 400

    def __init__(
        self,
        parent,
        var: tk.StringVar,
        on_change=None,
        *,
        notify_on_slide: bool = False,
        max_slider_steps: int | None = _DEFAULT_MAX_SLIDER_STEPS,
    ) -> None:
        self.var = var
        self.values: list[int] = []
        self._value_to_index: dict[int, int] = {}
        self._slider_steps = 1
        self._index_scale = 1.0
        self._max_slider_steps = max_slider_steps
        self._playback_mode = False
        self._updating = False
        self._on_change = on_change
        self._notify_on_slide = notify_on_slide

        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill=tk.X)
        row.grid_columnconfigure(3, weight=1)

        self._minus_btn = G.ghost_button(row, "−", lambda: self.step(-1), width=36)
        self._minus_btn.grid(row=0, column=0, padx=(0, 4))
        self.entry = G.entry_field(row, var, width=88)
        self.entry.grid(row=0, column=1, padx=(0, 4))
        self.entry.bind("<Return>", self._on_entry_commit)
        self.entry.bind("<FocusOut>", self._on_entry_commit)
        self._plus_btn = G.ghost_button(row, "+", lambda: self.step(1), width=36)
        self._plus_btn.grid(row=0, column=2, padx=(0, 8))

        self.slider = ctk.CTkSlider(
            row,
            from_=0,
            to=1,
            number_of_steps=1,
            command=self._on_slide,
            button_color=C["accent"],
            progress_color=C["accent"],
            button_hover_color=C["accent_hover"],
        )
        self.slider.grid(row=0, column=3, sticky="ew")
        self.slider.set(0)
        self.slider.bind("<ButtonRelease-1>", self._on_slide_release)

    def step(self, delta: int) -> None:
        if not self.values or self._playback_mode:
            return
        current = self._parse_current()
        if current is not None and current in self._value_to_index:
            idx = self._value_to_index[current]
        else:
            idx = int(round(float(self.slider.get()) * self._index_scale))
        self._set_index(idx + delta, notify=True)

    def _on_slide(self, _value: float) -> None:
        if self._updating or not self.values or self._playback_mode:
            return
        idx = int(round(float(self.slider.get()) * self._index_scale))
        self._set_index(idx, notify=self._notify_on_slide)

    def _on_slide_release(self, _event=None) -> None:
        if self._updating or not self.values or not self._on_change or self._playback_mode:
            return
        self._on_change()

    def _on_entry_commit(self, _event=None) -> None:
        if self._updating or not self.values or self._playback_mode:
            return
        parsed = self._parse_current()
        if parsed is None:
            idx = int(round(float(self.slider.get()) * self._index_scale))
            self._set_index(idx, notify=False)
            return
        idx = self._value_to_index[self._snap_value(parsed)]
        self._set_index(idx, notify=True)

    def _snap_value(self, raw: int) -> int:
        if raw in self.values:
            return raw
        return min(self.values, key=lambda v: abs(v - raw))

    def _set_index(self, idx: int, *, notify: bool) -> None:
        if not self.values:
            return
        idx = max(0, min(len(self.values) - 1, idx))
        self._updating = True
        if len(self.values) <= 1:
            slider_pos = 0.0
        else:
            slider_pos = idx / (len(self.values) - 1) * self._slider_steps
        self.slider.set(slider_pos)
        self.var.set(str(self.values[idx]))
        self._updating = False
        if notify and self._on_change:
            self._on_change()

    def configure(
        self,
        values: list[int],
        *,
        pick: str | int = "last",
    ) -> None:
        self.values = list(values)
        self._value_to_index = {value: i for i, value in enumerate(self.values)}
        self._updating = True
        if not self.values:
            self.var.set("")
            self.slider.configure(state="disabled")
            self.entry.configure(state="disabled")
            self._minus_btn.configure(state="disabled")
            self._plus_btn.configure(state="disabled")
            self._updating = False
            return

        n = len(self.values)
        cap = self._max_slider_steps
        if cap is None:
            self._slider_steps = max(n - 1, 1)
        else:
            self._slider_steps = min(max(n - 1, 1), cap)
        self._index_scale = (n - 1) / self._slider_steps if self._slider_steps else 1.0
        self.slider.configure(
            state="normal",
            from_=0,
            to=self._slider_steps,
            number_of_steps=self._slider_steps,
        )
        self.entry.configure(state="normal")
        self._minus_btn.configure(state="normal")
        self._plus_btn.configure(state="normal")

        current = self._parse_current()
        if pick == "keep" and current is not None and self.values:
            idx = self._index_for_value(current)
        elif pick == "first":
            idx = 0
        elif pick == "mid":
            idx = n // 2
        elif pick == "last":
            idx = n - 1
        elif isinstance(pick, int) and pick in self._value_to_index:
            idx = self._value_to_index[pick]
        else:
            idx = n - 1
        self._updating = False
        self._set_index(idx, notify=False)

    def _index_for_value(self, value: int) -> int:
        if value in self._value_to_index:
            return self._value_to_index[value]
        return self._value_to_index[self._snap_value(value)]

    def set_value(self, value: int) -> None:
        if value in self._value_to_index:
            self._set_index(self._value_to_index[value], notify=False)

    def set_display_only(self, value: int) -> None:
        """Update the entry during playback without moving the slider thumb."""
        if value not in self._value_to_index:
            return
        self._updating = True
        self.var.set(str(value))
        self._updating = False

    def sync_slider(self) -> None:
        """Move the slider thumb to match the current entry value."""
        current = self._parse_current()
        if current is None or not self.values:
            return
        self._set_index(self._index_for_value(current), notify=False)

    def set_playback_mode(self, active: bool) -> None:
        """Lock the slider while frames advance automatically."""
        self._playback_mode = active
        state = "disabled" if active else "normal"
        self.slider.configure(state=state)
        self._minus_btn.configure(state=state)
        self._plus_btn.configure(state=state)

    def _parse_current(self) -> int | None:
        try:
            return int(self.var.get().strip())
        except (TypeError, ValueError):
            return None


class MdsViewApp(ctk.CTk):
    _NAV = (
        ("browse", "◫", "Catalog"),
        ("plot", "◉", "Field"),
        ("grid", "⊞", "Grid"),
        ("diff", "△", "Diff"),
        ("dod", "◎", "DiD"),
    )

    def __init__(self, initial_dir: str = ".") -> None:
        super().__init__()
        G.init_app()

        self.title(G.APP_NAME)
        self.geometry("1360x900")
        self.minsize(1100, 760)
        self.configure(fg_color=C["bg"])

        self.data_dir = tk.StringVar(value=os.path.abspath(initial_dir))
        self.status = tk.StringVar(value="Open a run directory")
        self.run_summary = tk.StringVar(value="")
        self.view_caption = tk.StringVar(value="No field loaded")
        self.view_meta = tk.StringVar(value="Select a variable from the catalog")

        self.plot_var = tk.StringVar()
        self.plot_iter = tk.StringVar()
        self.plot_level = tk.StringVar(value="0")
        self.plot_cmap = tk.StringVar(value=DEFAULT_PLOT_CMAP)
        self.plot_vmin = tk.StringVar()
        self.plot_vmax = tk.StringVar()

        self.diff_later_dir = tk.StringVar(value=os.path.abspath(initial_dir))
        self.diff_earlier_dir = tk.StringVar(value=os.path.abspath(initial_dir))
        self.diff_var = tk.StringVar()
        self.diff_later = tk.StringVar()
        self.diff_earlier = tk.StringVar()
        self.diff_level = tk.StringVar(value="0")
        self.diff_cmap = tk.StringVar(value=DEFAULT_DIFF_CMAP)
        self.diff_vmin = tk.StringVar()
        self.diff_vmax = tk.StringVar()

        self.dod_a = tk.StringVar()
        self.dod_b = tk.StringVar()
        self.dod_t1 = tk.StringVar()
        self.dod_t2 = tk.StringVar()
        self.dod_level = tk.StringVar(value="0")
        self.dod_cmap = tk.StringVar(value=DEFAULT_DIFF_CMAP)
        self.dod_vmin = tk.StringVar()
        self.dod_vmax = tk.StringVar()

        self.grid_coord_mode = tk.StringVar(value="centers")
        self.grid_overlay = tk.BooleanVar(value=False)

        self.stat_min = tk.StringVar(value="—")
        self.stat_mean = tk.StringVar(value="—")
        self.stat_max = tk.StringVar(value="—")
        self.stat_std = tk.StringVar(value="—")

        self.playback_fps = tk.DoubleVar(value=2.0)
        self.playback_loop = tk.BooleanVar(value=True)
        self.playback_lock_scale = tk.BooleanVar(value=True)

        self._prefixes: list[str] = []
        self._catalog: catalog.RunCatalog | None = None
        self._iter_map: dict[str, list[int]] = {}
        self._iter_sets: dict[str, set[int]] = {}
        self._shape_cache: dict[tuple[str, str], tuple[int, ...]] = {}
        self._diff_iter_map: dict[tuple[str, str], list[int]] = {}
        self._diff_iter_sets: dict[tuple[str, str], set[int]] = {}
        self._playing = False
        self._playback_job: str | None = None
        self._auto_plot_job: str | None = None
        self._playback_index = 0
        self._playback_iters: list[int] = []
        self._playback_vmin: float | None = None
        self._playback_vmax: float | None = None
        self._playback_clim_key: tuple | None = None
        self._playback_anchor_iter: int | None = None
        self._play_btn: ctk.CTkButton | None = None
        self._nav_buttons: dict[str, ctk.CTkButton] = {}
        self._panels: dict[str, ctk.CTkFrame | ctk.CTkScrollableFrame] = {}
        self._active_nav = "browse"
        self._closing = False

        self._build_ui()
        self._bind_shortcuts()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.refresh_directory()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        self._build_status()
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self._build_topbar(body)

        main = ctk.CTkFrame(body, fg_color="transparent")
        main.pack(fill=tk.BOTH, expand=True, padx=14, pady=(0, 10))
        main.grid_columnconfigure(2, weight=1)
        main.grid_rowconfigure(0, weight=1)

        self._build_nav(main)
        self._build_controls(main)
        self._build_viewer(main)

    def _build_status(self) -> None:
        bar = ctk.CTkFrame(self, fg_color=C["status_bg"], corner_radius=0, height=40)
        bar.pack(side=tk.BOTTOM, fill=tk.X)
        bar.pack_propagate(False)
        ctk.CTkLabel(
            bar, textvariable=self.status, font=G.FONTS["caption"], text_color=C["muted"], anchor="w"
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=16)
        ctk.CTkLabel(
            bar, textvariable=self.run_summary, font=G.FONTS["caption"], text_color=C["text"], anchor="e"
        ).pack(side=tk.RIGHT, padx=16)

    def _build_topbar(self, parent) -> None:
        bar = ctk.CTkFrame(parent, fg_color=C["surface"], corner_radius=0, height=72)
        bar.pack(fill=tk.X)
        bar.pack_propagate(False)

        inner = ctk.CTkFrame(bar, fg_color="transparent")
        inner.pack(fill=tk.BOTH, expand=True, padx=16, pady=10)
        inner.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(inner, text=G.APP_NAME, font=G.FONTS["brand"], text_color=C["text"]).grid(
            row=0, column=0, sticky="w", padx=(0, 20)
        )
        G.entry_field(inner, self.data_dir).grid(row=0, column=1, sticky="ew", padx=(0, 10))
        btns = ctk.CTkFrame(inner, fg_color="transparent")
        btns.grid(row=0, column=2, sticky="e")
        G.ghost_button(btns, "Open", self.browse_directory).pack(side=tk.LEFT, padx=3)
        G.ghost_button(btns, "Refresh", self.refresh_directory).pack(side=tk.LEFT, padx=3)
        G.ghost_button(btns, "PNG", self.save_png).pack(side=tk.LEFT, padx=3)
        G.ghost_button(btns, "GIF", self.show_gif_dialog).pack(side=tk.LEFT, padx=3)

    def _build_nav(self, parent) -> None:
        rail = ctk.CTkFrame(
            parent,
            fg_color=C["nav"],
            width=68,
            corner_radius=G.CTK["corner_radius"],
            border_color=C["border"],
            border_width=G.CTK["border_width"],
        )
        rail.grid(row=0, column=0, sticky="ns", padx=(0, 10))
        rail.grid_propagate(False)

        for key, symbol, _tip in self._NAV:
            btn = G.nav_button(rail, symbol, lambda k=key: self._select_nav(k))
            btn.pack(pady=4, padx=6)
            self._nav_buttons[key] = btn

    def _build_controls(self, parent) -> None:
        shell = ctk.CTkFrame(
            parent,
            fg_color=C["surface"],
            width=320,
            corner_radius=G.CTK["corner_radius"],
            border_color=C["border_strong"],
            border_width=G.CTK["border_width"],
        )
        shell.grid(row=0, column=1, sticky="ns", padx=(0, 12))
        shell.grid_propagate(False)

        self._control_title = ctk.CTkLabel(
            shell, text="Catalog", font=G.FONTS["heading"], text_color=C["text"], anchor="w"
        )
        self._control_title.pack(fill=tk.X, padx=16, pady=(14, 8))

        for key, _s, title in self._NAV:
            if key == "browse":
                panel: ctk.CTkFrame | ctk.CTkScrollableFrame = ctk.CTkScrollableFrame(
                    shell,
                    fg_color="transparent",
                    scrollbar_button_color=C["border"],
                    scrollbar_button_hover_color=C["accent_soft"],
                )
            else:
                panel = ctk.CTkFrame(shell, fg_color="transparent")
            self._panels[key] = panel
            setattr(self, f"_{key}_panel", panel)

        self._build_browse_panel()
        self._build_plot_panel()
        self._build_grid_panel()
        self._build_diff_panel()
        self._build_dod_panel()
        self._select_nav("browse")

    def _build_viewer(self, parent) -> None:
        card = G.card_frame(parent)
        card.grid(row=0, column=2, sticky="nsew")
        card.grid_columnconfigure(0, weight=1)
        card.grid_rowconfigure(2, weight=1)

        head = ctk.CTkFrame(card, fg_color="transparent")
        head.grid(row=0, column=0, sticky="ew", padx=18, pady=(16, 8))
        head.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            head, textvariable=self.view_caption, font=G.FONTS["heading"], text_color=C["text"], anchor="w"
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(
            head, textvariable=self.view_meta, font=G.FONTS["caption"], text_color=C["muted"], anchor="w"
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))

        stats = ctk.CTkFrame(
            card,
            fg_color=C["stat_bg"],
            corner_radius=8,
            border_color=C["border"],
            border_width=G.CTK["border_width"],
        )
        stats.grid(row=1, column=0, sticky="ew", padx=18, pady=(0, 10))
        for col, (label, var) in enumerate(
            (("Minimum", self.stat_min), ("Mean", self.stat_mean), ("Maximum", self.stat_max), ("Std dev", self.stat_std))
        ):
            cell = ctk.CTkFrame(stats, fg_color="transparent")
            cell.grid(row=0, column=col, sticky="ew", padx=14, pady=8)
            stats.grid_columnconfigure(col, weight=1)
            ctk.CTkLabel(cell, text=label, font=G.FONTS["caption"], text_color=C["muted"]).pack(anchor="w")
            ctk.CTkLabel(cell, textvariable=var, font=G.FONTS["stat"], text_color=C["text"]).pack(anchor="w")

        plot_host = ctk.CTkFrame(
            card,
            fg_color=C["axes_bg"],
            corner_radius=8,
            border_color=C["border_strong"],
            border_width=G.CTK["border_width"],
        )
        plot_host.grid(row=2, column=0, sticky="nsew", padx=18, pady=(0, 10))
        plot_host.grid_columnconfigure(0, weight=1)
        plot_host.grid_rowconfigure(0, weight=1)

        self.fig, self.ax = plt.subplots(figsize=(6.8, 4.2), facecolor=C["figure_bg"])
        self.fig.set_dpi(100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_host)
        w = self.canvas.get_tk_widget()
        w.configure(bg=C["axes_bg"], highlightthickness=0)
        w.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

        self._toolbar_host = ctk.CTkFrame(plot_host, fg_color="transparent", height=1)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self._toolbar_host, pack_toolbar=False)

        footer = ctk.CTkFrame(card, fg_color="transparent")
        footer.grid(row=3, column=0, sticky="ew", padx=18, pady=(0, 14))

        tools = ctk.CTkFrame(footer, fg_color="transparent")
        tools.pack(side=tk.LEFT)
        G.ghost_button(tools, "Reset", self.toolbar.home, width=56).pack(side=tk.LEFT, padx=(0, 4))
        G.ghost_button(tools, "Pan", self.toolbar.pan, width=56).pack(side=tk.LEFT, padx=4)
        G.ghost_button(tools, "Zoom", self.toolbar.zoom, width=56).pack(side=tk.LEFT, padx=4)

        transport = ctk.CTkFrame(footer, fg_color="transparent")
        transport.pack(side=tk.LEFT, padx=(20, 0))
        for text, cmd in (("⏮", lambda: self.jump_playback_frame(0)), ("◀", lambda: self.step_playback_frame(-1))):
            G.ghost_button(transport, text, cmd, width=36).pack(side=tk.LEFT, padx=2)
        self._play_btn = G.accent_button(transport, "Play", self.toggle_playback, width=72)
        self._play_btn.pack(side=tk.LEFT, padx=6)
        for text, cmd in (("▶", lambda: self.step_playback_frame(1)), ("⏭", lambda: self.jump_playback_frame(-1))):
            G.ghost_button(transport, text, cmd, width=36).pack(side=tk.LEFT, padx=2)
        G.ghost_button(transport, "Stop", self.stop_playback, width=56).pack(side=tk.LEFT, padx=(6, 12))
        ctk.CTkCheckBox(
            transport, text="Loop", variable=self.playback_loop, font=G.FONTS["caption"], text_color=C["text"]
        ).pack(side=tk.LEFT, padx=(0, 8))
        ctk.CTkLabel(transport, text="FPS", font=G.FONTS["caption"], text_color=C["muted"]).pack(side=tk.LEFT)
        ctk.CTkEntry(
            transport,
            textvariable=self.playback_fps,
            width=48,
            height=28,
            font=G.FONTS["caption"],
            fg_color=C["card"],
            border_color=C["border_strong"],
            border_width=G.CTK["border_width"],
        ).pack(side=tk.LEFT, padx=(4, 0))

        self._show_welcome()

    def _select_nav(self, key: str) -> None:
        self._active_nav = key
        titles = {k: t for k, _s, t in self._NAV}
        self._control_title.configure(text=titles[key])
        for name, panel in self._panels.items():
            if name == key:
                panel.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))
            else:
                panel.pack_forget()
        for name, btn in self._nav_buttons.items():
            if name == key:
                btn.configure(
                    fg_color=C["nav_active"],
                    text_color=C["accent"],
                    border_width=G.CTK["border_width"],
                    border_color=C["border_strong"],
                )
            else:
                btn.configure(fg_color="transparent", text_color=C["muted"], border_width=0)

    def _build_browse_panel(self) -> None:
        p = self._panels["browse"]
        G.section_label(p, "Variables").pack(anchor="w", pady=(0, 6))
        list_wrap = ctk.CTkFrame(
            p,
            fg_color=C["card"],
            corner_radius=8,
            border_width=G.CTK["border_width"],
            border_color=C["border_strong"],
        )
        list_wrap.pack(fill=tk.BOTH, expand=True)
        self.var_list = tk.Listbox(list_wrap, height=8, exportselection=False, relief=tk.FLAT)
        G.style_listbox(self.var_list)
        self.var_list.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        self.var_list.bind("<<ListboxSelect>>", self._on_var_select)
        self.var_list.bind("<Double-Button-1>", lambda _e: self._send_to_plot())

        G.section_label(p, "Metadata").pack(anchor="w", pady=(12, 6))
        self.detail_text = ctk.CTkTextbox(
            p,
            height=180,
            font=G.FONTS["mono"],
            fg_color=C["surface"],
            text_color=C["text"],
            border_color=C["border_strong"],
            border_width=G.CTK["border_width"],
        )
        self.detail_text.pack(fill=tk.BOTH, expand=True)
        self.detail_text.configure(state="disabled")

        row = ctk.CTkFrame(p, fg_color="transparent")
        row.pack(fill=tk.X, pady=(12, 0))
        G.accent_button(row, "Plot selected", self._send_to_plot).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        G.ghost_button(row, ".meta", self._show_raw_meta, width=64).pack(side=tk.LEFT)

    def _build_plot_panel(self) -> None:
        p = self._panels["plot"]
        self._plot_card = p
        self._field_row(p, "Variable", self.plot_var, self._on_plot_var_change, "plot_var_combo")
        self._slider_row(
            p, "Iteration", self.plot_iter, "plot_iter_slider",
            on_change=self._schedule_auto_plot, max_slider_steps=None,
        )
        self._slider_row(
            p, "Level", self.plot_level, "plot_level_slider",
            on_change=self._schedule_auto_plot,
        )
        self._cmap_limits_row(
            p,
            self.plot_cmap,
            self.plot_vmin,
            self.plot_vmax,
            "plot_cmap_combo",
            self._schedule_auto_plot,
        )

    def _build_grid_panel(self) -> None:
        p = self._panels["grid"]
        self._grid_card = p

        G.section_label(p, "Map coordinates").pack(anchor="w", pady=(0, 4))
        for key, label in grid.COORD_MODE_LABELS.items():
            ctk.CTkRadioButton(
                p,
                text=label,
                variable=self.grid_coord_mode,
                value=key,
                font=G.FONTS["label"],
                text_color=C["text"],
                command=self._on_grid_settings_change,
            ).pack(anchor="w", pady=2)

        ctk.CTkCheckBox(
            p,
            text="Overlay grid lines on field plots",
            variable=self.grid_overlay,
            font=G.FONTS["label"],
            text_color=C["text"],
            command=self._on_grid_settings_change,
        ).pack(anchor="w", pady=(12, 8))

        G.section_label(p, "Run grid files").pack(anchor="w", pady=(8, 4))
        self._grid_info_text = ctk.CTkTextbox(
            p,
            height=120,
            font=G.FONTS["mono"],
            fg_color=C["surface"],
            text_color=C["text"],
            border_color=C["border_strong"],
            border_width=G.CTK["border_width"],
        )
        self._grid_info_text.pack(fill=tk.X, pady=(0, 10))
        self._grid_info_text.configure(state="disabled")

        G.section_label(p, "About").pack(anchor="w", pady=(4, 4))
        ctk.CTkLabel(
            p,
            text=grid.GRID_HELP,
            font=G.FONTS["caption"],
            text_color=C["muted"],
            justify="left",
            wraplength=280,
        ).pack(anchor="w", pady=(0, 12))

        G.accent_button(p, "Preview grid", self.plot_grid_preview).pack(fill=tk.X, pady=(0, 6))
        G.ghost_button(p, "Apply to current field", self._apply_grid_to_field).pack(fill=tk.X)

    def _build_diff_panel(self) -> None:
        p = self._panels["diff"]
        self._diff_card = p
        self._dir_picker_row(p, "Later run", self.diff_later_dir, self._on_diff_dirs_change)
        self._dir_picker_row(p, "Earlier run", self.diff_earlier_dir, self._on_diff_dirs_change)
        self._field_row(p, "Variable", self.diff_var, self._on_diff_var_change, "diff_var_combo")
        self._slider_row(p, "Later", self.diff_later, "diff_later_slider", on_change=None, max_slider_steps=None)
        self._slider_row(p, "Earlier", self.diff_earlier, "diff_earlier_slider", on_change=None, max_slider_steps=None)
        self._slider_row(p, "Level", self.diff_level, "diff_level_slider", on_change=None)
        self._cmap_limits_row(p, self.diff_cmap, self.diff_vmin, self.diff_vmax, "diff_cmap_combo")
        G.accent_button(p, "Plot difference", self.plot_diff).pack(fill=tk.X, pady=(14, 6))
        G.ghost_button(p, "Save MDS", self.save_diff_field).pack(fill=tk.X)

    def _build_dod_panel(self) -> None:
        p = self._panels["dod"]
        self._dod_card = p
        self._field_row(p, "Variable A", self.dod_a, self._fill_time_combos, "dod_a_combo")
        self._field_row(p, "Variable B", self.dod_b, self._fill_time_combos, "dod_b_combo")
        self._slider_row(p, "Time t1", self.dod_t1, "dod_t1_slider", on_change=None, max_slider_steps=None)
        self._slider_row(p, "Time t2", self.dod_t2, "dod_t2_slider", on_change=None, max_slider_steps=None)
        self._slider_row(p, "Level", self.dod_level, "dod_level_slider", on_change=None)
        self._cmap_limits_row(p, self.dod_cmap, self.dod_vmin, self.dod_vmax, "dod_cmap_combo")
        G.accent_button(p, "Plot DiD", self.plot_dod).pack(fill=tk.X, pady=(14, 6))
        G.ghost_button(p, "Volume stats", self.show_dod_stats).pack(fill=tk.X, pady=(0, 6))
        G.ghost_button(p, "Save MDS", self.save_dod_field).pack(fill=tk.X)

    def _field_row(self, parent, label: str, var: tk.StringVar, command, attr: str) -> None:
        G.section_label(parent, label).pack(anchor="w", pady=(10, 4))
        menu = G.option_menu(parent, var, [], command=lambda _v: command() if command else None)
        menu.pack(fill=tk.X)
        setattr(parent, attr, menu)

    def _menu_row(
        self, parent, label: str, var: tk.StringVar, values: list[str], attr: str, command=None
    ) -> None:
        G.section_label(parent, label).pack(anchor="w", pady=(10, 4))
        menu = G.option_menu(parent, var, values, command=command)
        menu.pack(fill=tk.X)
        setattr(parent, attr, menu)

    def _slider_row(
        self,
        parent,
        label: str,
        var: tk.StringVar,
        attr: str,
        on_change=None,
        *,
        notify_on_slide: bool = False,
        max_slider_steps: int | None = _IndexSlider._DEFAULT_MAX_SLIDER_STEPS,
    ) -> _IndexSlider:
        G.section_label(parent, label).pack(anchor="w", pady=(10, 4))
        slider = _IndexSlider(
            parent, var, on_change=on_change,
            notify_on_slide=notify_on_slide, max_slider_steps=max_slider_steps,
        )
        setattr(parent, attr, slider)
        return slider

    def _dir_picker_row(
        self, parent, label: str, var: tk.StringVar, on_change=None
    ) -> None:
        G.section_label(parent, label).pack(anchor="w", pady=(10, 4))
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill=tk.X)
        row.grid_columnconfigure(0, weight=1)
        entry = ctk.CTkEntry(
            row,
            textvariable=var,
            font=G.FONTS["label"],
            fg_color=C["card"],
            border_color=C["border_strong"],
            border_width=G.CTK["border_width"],
            text_color=C["text"],
            corner_radius=8,
            height=G.CTK["entry_height"],
        )
        entry.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        def browse() -> None:
            path = filedialog.askdirectory(initialdir=var.get() or self.data_dir.get())
            if path:
                var.set(os.path.abspath(path))
                if on_change:
                    on_change()

        G.ghost_button(row, "…", browse, width=36).grid(row=0, column=1)

        def on_commit(_event=None) -> None:
            if on_change:
                on_change()

        entry.bind("<FocusOut>", on_commit)
        entry.bind("<Return>", on_commit)

    def _cmap_limits_row(
        self,
        parent,
        cmap_var: tk.StringVar,
        vmin_var: tk.StringVar,
        vmax_var: tk.StringVar,
        cmap_attr: str,
        command=None,
    ) -> None:
        G.section_label(parent, "Colormap · limits").pack(anchor="w", pady=(10, 4))
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill=tk.X)
        row.grid_columnconfigure(0, weight=1)
        menu = G.option_menu(row, cmap_var, COLORMAP_CHOICES, command=command)
        menu.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        G.entry_field(row, vmin_var, width=64).grid(row=0, column=1, padx=(0, 4))
        ctk.CTkLabel(row, text="to", font=G.FONTS["caption"], text_color=C["muted"]).grid(
            row=0, column=2, padx=2
        )
        G.entry_field(row, vmax_var, width=64).grid(row=0, column=3)
        setattr(parent, cmap_attr, menu)

    def _limits_row(self, parent, vmin_var: tk.StringVar, vmax_var: tk.StringVar) -> None:
        G.section_label(parent, "Colour limits").pack(anchor="w", pady=(10, 4))
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill=tk.X)
        G.entry_field(row, vmin_var, width=90).pack(side=tk.LEFT, padx=(0, 8))
        ctk.CTkLabel(row, text="to", font=G.FONTS["caption"], text_color=C["muted"]).pack(side=tk.LEFT, padx=4)
        G.entry_field(row, vmax_var, width=90).pack(side=tk.LEFT)

    def _bind_shortcuts(self) -> None:
        self.bind("<Control-o>", lambda _e: self.browse_directory())
        self.bind("<Control-r>", lambda _e: self.refresh_directory())
        self.bind("<Control-s>", lambda _e: self.save_png())
        self.bind("<space>", lambda _e: self.toggle_playback())

    def _schedule_auto_plot(self, _event=None) -> None:
        if self._closing:
            return
        if self._auto_plot_job:
            self.after_cancel(self._auto_plot_job)
        self._auto_plot_job = self.after(150, self._run_auto_plot)

    def _run_auto_plot(self) -> None:
        self._auto_plot_job = None
        if self._closing:
            return
        self._auto_plot_if_ready()

    def _auto_plot_if_ready(self) -> None:
        if self._closing:
            return
        if self._playing or self._active_nav != "plot":
            return
        if not self.plot_var.get() or not self.plot_iter.get():
            return
        if self._safe_level(self.plot_var.get(), self.plot_level) is None:
            return
        try:
            self.plot_snapshot()
        except Exception:
            pass

    def _show_welcome(self) -> None:
        self._clear_figure()
        self.fig.patch.set_facecolor(C["figure_bg"])
        self.ax.set_facecolor(C["figure_bg"])
        self.ax.text(
            0.5, 0.55, G.APP_NAME, transform=self.ax.transAxes, ha="center",
            fontsize=28, color=C["muted"], fontfamily="sans-serif", weight="bold",
        )
        self.ax.text(
            0.5, 0.38, "MITgcm MDS viewer", transform=self.ax.transAxes, ha="center",
            fontsize=12, color=C["muted"], fontfamily="sans-serif",
        )
        self.ax.axis("off")
        self.canvas.draw()
        self._reset_stats()
        self.view_caption.set("No field loaded")
        self.view_meta.set("Open a run folder, then choose a field")

    def _reset_stats(self) -> None:
        for var in (self.stat_min, self.stat_mean, self.stat_max, self.stat_std):
            var.set("—")

    def _update_field_stats(self, field2d: np.ndarray) -> None:
        finite = field2d[np.isfinite(field2d)]
        if finite.size == 0:
            self._reset_stats()
            return
        self.stat_min.set(f"{np.min(finite):.4g}")
        self.stat_mean.set(f"{np.mean(finite):.4g}")
        self.stat_max.set(f"{np.max(finite):.4g}")
        self.stat_std.set(f"{np.std(finite):.4g}")

    def _set_view(self, caption: str, meta: str) -> None:
        self.view_caption.set(caption)
        self.view_meta.set(meta)

    def _log(self, msg: str) -> None:
        self.status.set(msg)

    def _set_detail(self, text: str) -> None:
        self.detail_text.configure(state="normal")
        self.detail_text.delete("1.0", "end")
        self.detail_text.insert("1.0", text)
        self.detail_text.configure(state="disabled")

    def _level_indices_for_shape(self, shape: tuple[int, ...]) -> list[int]:
        _, nz = ops.level_axis(shape)
        return list(range(max(int(nz), 1)))

    def _configure_level_slider(
        self,
        card,
        var: tk.StringVar,
        attr: str,
        shape: tuple[int, ...],
    ) -> None:
        levels = self._level_indices_for_shape(shape)
        _, nz = ops.level_axis(shape)
        surface_only = nz <= 1
        slider = getattr(card, attr)
        slider.configure(levels, pick="first" if surface_only else "keep")
        if surface_only:
            slider.entry.configure(state="disabled")
            slider.slider.configure(state="disabled")
            slider._minus_btn.configure(state="disabled")
            slider._plus_btn.configure(state="disabled")
        elif levels:
            slider.entry.configure(state="normal")
            slider.slider.configure(state="normal")
            slider._minus_btn.configure(state="normal")
            slider._plus_btn.configure(state="normal")

    def _update_plot_level_slider(self, prefix: str | None = None) -> None:
        prefix = prefix or self.plot_var.get()
        if not prefix:
            return
        try:
            self._configure_level_slider(
                self._plot_card, self.plot_level, "plot_level_slider",
                self._shape_for(self.data_dir.get(), prefix),
            )
        except OSError:
            pass

    def _update_diff_level_slider(self, prefix: str | None = None) -> None:
        prefix = prefix or self.diff_var.get()
        if not prefix:
            return
        try:
            self._configure_level_slider(
                self._diff_card, self.diff_level, "diff_level_slider",
                self._shape_for(self.diff_later_dir.get(), prefix),
            )
        except OSError:
            pass

    def _update_dod_level_slider(self, prefix: str | None = None) -> None:
        prefix = prefix or self.dod_a.get() or self.plot_var.get()
        if not prefix:
            return
        try:
            self._configure_level_slider(
                self._dod_card, self.dod_level, "dod_level_slider",
                self._shape_for(self.data_dir.get(), prefix),
            )
        except OSError:
            pass

    def _combo_set_values(self, menu: ctk.CTkOptionMenu, values: list[str]) -> None:
        menu.configure(values=values or [""])
        if values:
            menu.set(values[-1])

    def _sync_combos(self) -> None:
        names = self._prefixes
        for menu in (
            self._plot_card.plot_var_combo,
            self._dod_card.dod_a_combo,
            self._dod_card.dod_b_combo,
        ):
            self._combo_set_values(menu, names)
        if names:
            self.plot_var.set("T" if "T" in names else names[0])
            self.dod_a.set("T" if "T" in names else names[0])
            self.dod_b.set("S" if "S" in names else (names[1] if len(names) > 1 else names[0]))
        self._on_plot_var_change()
        self._sync_diff_combos()
        self._fill_time_combos()

    def _fill_time_combos(self) -> None:
        prefix = self.dod_a.get() or self.plot_var.get()
        iters = self._iters_for(prefix)
        self._dod_card.dod_t1_slider.configure(iters, pick="first")
        self._dod_card.dod_t2_slider.configure(iters, pick="last")
        self._update_dod_level_slider(prefix)

    def _shape_for(self, data_dir: str, prefix: str) -> tuple[int, ...]:
        key = (os.path.abspath(data_dir), prefix)
        if key not in self._shape_cache:
            if self._catalog and os.path.abspath(data_dir) == self._catalog.data_dir:
                self._shape_cache[key] = self._catalog.shape(prefix)
            else:
                self._shape_cache[key] = io.field_info(data_dir, prefix).shape
        return self._shape_cache[key]

    def _has_iteration(self, prefix: str, raw: str) -> bool:
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return False
        return value in self._iter_sets.get(prefix, set())

    def _iters_for(self, prefix: str) -> list[int]:
        if not prefix:
            return []
        if prefix not in self._iter_map:
            if self._catalog:
                iters = self._catalog.iters_for(prefix)
            else:
                iters = io.list_iterations(self.data_dir.get(), prefix)
            self._iter_map[prefix] = iters
            self._iter_sets[prefix] = set(iters)
        return self._iter_map[prefix]

    def _iters_for_dir(self, data_dir: str, prefix: str) -> list[int]:
        if not prefix or not data_dir:
            return []
        key = (os.path.abspath(data_dir), prefix)
        if key not in self._diff_iter_map:
            if self._catalog and key[0] == self._catalog.data_dir:
                iters = self._catalog.iters_for(prefix)
            else:
                iters = io.list_iterations(data_dir, prefix)
            self._diff_iter_map[key] = iters
            self._diff_iter_sets[key] = set(iters)
        return self._diff_iter_map[key]

    def _diff_prefixes(self) -> list[str]:
        later_dir = self.diff_later_dir.get()
        earlier_dir = self.diff_earlier_dir.get()
        try:
            later = set(io.list_prefixes(later_dir))
            earlier = set(io.list_prefixes(earlier_dir))
        except OSError:
            return []
        if later_dir == earlier_dir:
            return sorted(later)
        common = sorted(later & earlier)
        return common if common else sorted(later)

    def _sync_diff_combos(self) -> None:
        names = self._diff_prefixes()
        self._combo_set_values(self._diff_card.diff_var_combo, names)
        if names:
            current = self.diff_var.get()
            if current not in names:
                self.diff_var.set(names[0])
        self._on_diff_var_change()

    def _on_diff_dirs_change(self) -> None:
        self._diff_iter_map.clear()
        self._diff_iter_sets.clear()
        self._sync_diff_combos()

    # ---------------------------------------------------------------- data
    def browse_directory(self) -> None:
        path = filedialog.askdirectory(initialdir=self.data_dir.get())
        if path:
            self.data_dir.set(path)
            self.refresh_directory()

    def refresh_directory(self) -> None:
        data_dir = os.path.abspath(self.data_dir.get())
        self.data_dir.set(data_dir)
        self.diff_later_dir.set(data_dir)
        self.diff_earlier_dir.set(data_dir)
        self._iter_map.clear()
        self._iter_sets.clear()
        self._diff_iter_map.clear()
        self._diff_iter_sets.clear()
        self._shape_cache.clear()
        self._playback_clim_key = None
        try:
            catalog.invalidate_catalog(data_dir)
            self._catalog = catalog.get_catalog(data_dir)
            self._prefixes = self._catalog.prefixes
        except OSError as exc:
            messagebox.showerror("Error", str(exc))
            return

        self.var_list.delete(0, tk.END)
        for prefix in self._prefixes:
            try:
                summary = self._catalog.meta_summary(prefix)
                n_iters = self._catalog.iter_count(prefix)
                line = (
                    f"{prefix:8s}  {str(summary.shape):18s}  "
                    f"{summary.nbytes_human:>8s}  {n_iters} snaps"
                )
            except OSError:
                line = prefix
            self.var_list.insert(tk.END, line)

        if self._prefixes:
            self.var_list.selection_set(0)
            self._on_var_select()
            self._sync_combos()
            n = len(self._iters_for(self._prefixes[0]))
            self.run_summary.set(f"{len(self._prefixes)} fields · {n} snapshots · {os.path.basename(data_dir)}")
            self._log(f"Loaded {len(self._prefixes)} fields")
            self._select_nav("plot")
            self.after(200, self._auto_plot_if_ready)
        else:
            self._set_detail("No .meta files found.")
            self.run_summary.set("")
            self._show_welcome()
            self._log("No MDS output in folder")
        self._refresh_grid_info()

    def _grid_plot_kwargs(self) -> dict:
        return {
            "coord_mode": self.grid_coord_mode.get(),
            "overlay_grid": self.grid_overlay.get(),
        }

    def _refresh_grid_info(self) -> None:
        if not hasattr(self, "_grid_info_text"):
            return
        try:
            cat = grid.catalog_grid(self.data_dir.get())
            lines = cat.summary_lines()
            mode = self.grid_coord_mode.get()
            resolved = grid.resolve_coord_mode(mode, cat)
            lines.append("")
            lines.append(f"Selected: {grid.COORD_MODE_LABELS.get(mode, mode)}")
            if resolved != mode:
                lines.append(f"  → will use {grid.COORD_MODE_LABELS.get(resolved, resolved)}")
        except OSError as exc:
            lines = [str(exc)]
        self._grid_info_text.configure(state="normal")
        self._grid_info_text.delete("1.0", "end")
        self._grid_info_text.insert("1.0", "\n".join(lines))
        self._grid_info_text.configure(state="disabled")

    def _on_grid_settings_change(self) -> None:
        self._refresh_grid_info()
        if self._active_nav == "plot" and self.plot_var.get():
            self._schedule_auto_plot()

    def _apply_grid_to_field(self) -> None:
        self._select_nav("plot")
        self._schedule_auto_plot()

    def plot_grid_preview(self) -> None:
        data_dir = self.data_dir.get()
        mode = self.grid_coord_mode.get()
        try:
            self._clear_figure()
            self.fig.patch.set_facecolor(C["figure_bg"])
            self.ax.set_facecolor(C["figure_bg"])
            grid.plot_grid_preview(data_dir, mode=mode, ax=self.ax, show=False)
            self.canvas.draw()
            self._reset_stats()
            self._set_view(
                "Grid preview",
                grid.COORD_MODE_LABELS.get(mode, mode),
            )
            self._log("Grid preview")
        except Exception as exc:
            messagebox.showerror("Grid preview failed", str(exc))

    def _on_var_select(self, _event=None) -> None:
        sel = self.var_list.curselection()
        if not sel:
            return
        prefix = self._prefixes[sel[0]]
        try:
            summary = self._catalog.meta_summary(prefix) if self._catalog else io.read_field_meta(
                self.data_dir.get(), prefix, None
            )
            iters = self._iters_for(prefix)
            lines = [
                f"Variable   {prefix}",
                f"Shape      {summary.shape}",
                f"Precision  {summary.dataprec}",
                f"Size       {summary.nbytes_human}",
                f"Records    {summary.nrecords}",
                f"Iterations {iters[:12]}{' …' if len(iters) > 12 else ''}",
            ]
            self._set_detail("\n".join(lines))
        except OSError as exc:
            self._set_detail(str(exc))

    def _on_plot_var_change(self) -> None:
        prefix = self.plot_var.get()
        iters = self._iters_for(prefix)
        self._plot_card.plot_iter_slider.configure(iters, pick="first")
        self._update_plot_level_slider(prefix)
        self._schedule_auto_plot()

    def _on_diff_var_change(self) -> None:
        prefix = self.diff_var.get()
        later_dir = self.diff_later_dir.get()
        earlier_dir = self.diff_earlier_dir.get()
        later_iters = self._iters_for_dir(later_dir, prefix)
        earlier_iters = self._iters_for_dir(earlier_dir, prefix)
        self._diff_card.diff_later_slider.configure(later_iters, pick="last")
        self._diff_card.diff_earlier_slider.configure(earlier_iters, pick="first")
        self._update_diff_level_slider(prefix)

    @staticmethod
    def _short_dir(path: str) -> str:
        path = os.path.abspath(path)
        return os.path.basename(path) or path

    def _current_level(self, var: tk.StringVar) -> int | None:
        return self._parse_int(var.get())

    def _safe_level(self, prefix: str, level_var: tk.StringVar, data_dir: str | None = None) -> int | None:
        """Return a level index valid for prefix, clamping or resetting for 2-D fields."""
        level = self._current_level(level_var)
        if level is None or not prefix:
            return None
        data_dir = data_dir or self.data_dir.get()
        try:
            shape = self._shape_for(data_dir, prefix)
        except OSError:
            return level
        _, nz = ops.level_axis(shape)
        if nz <= 1:
            return 0
        return max(0, min(nz - 1, level))

    def _field_has_levels(self, prefix: str, data_dir: str | None = None) -> bool:
        try:
            _, nz = ops.level_axis(self._shape_for(data_dir or self.data_dir.get(), prefix))
            return nz > 1
        except OSError:
            return True

    def _playback_clim_key_for(
        self, prefix: str, iters: list[int], level: int, anchor_iteration: int | None = None,
    ) -> tuple:
        return (
            self.data_dir.get(),
            prefix,
            level,
            len(iters),
            iters[0] if iters else None,
            iters[-1] if iters else None,
            anchor_iteration,
        )

    def _ensure_playback_clim(
        self, prefix: str, iters: list[int], level: int, *, anchor_iteration: int,
    ) -> None:
        """Lock playback colour limits to the reference frame (or user vmin/vmax)."""
        if not self.playback_lock_scale.get():
            self._playback_clim_key = None
            return
        key = self._playback_clim_key_for(prefix, iters, level, anchor_iteration)
        if key == self._playback_clim_key:
            return
        self._playback_vmin, self._playback_vmax = animation.resolve_playback_clim(
            self.data_dir.get(), prefix, iters, level,
            user_vmin=self._parse_float(self.plot_vmin.get()),
            user_vmax=self._parse_float(self.plot_vmax.get()),
            lock_scale=True,
            anchor_iteration=anchor_iteration,
        )
        self._playback_clim_key = key

    def _level_phrase(self, prefix: str, level: int, data_dir: str | None = None) -> str:
        if self._field_has_levels(prefix, data_dir):
            return f"  ·  level {level}"
        return ""

    def _send_to_plot(self) -> None:
        sel = self.var_list.curselection()
        if not sel:
            return
        self.plot_var.set(self._prefixes[sel[0]])
        self._select_nav("plot")
        self._on_plot_var_change()
        self.after(100, self.plot_snapshot)

    def _show_raw_meta(self) -> None:
        sel = self.var_list.curselection()
        if not sel:
            return
        prefix = self._prefixes[sel[0]]
        try:
            info = io.field_info(self.data_dir.get(), prefix)
            summary = io.read_field_meta(
                self.data_dir.get(), prefix, info.iterations[-1] if info.iterations else None
            )
            with open(summary.meta_path, encoding="utf-8", errors="replace") as f:
                self._set_detail(f.read())
        except OSError as exc:
            messagebox.showerror("Meta", str(exc))

    # ---------------------------------------------------------------- playback
    def _playback_delay_ms(self, _frame_idx: int) -> int:
        return max(int(1000 / max(float(self.playback_fps.get()), 0.1)), 1)

    def _playback_settings(self) -> tuple[str, int, list[int]] | None:
        prefix = self.plot_var.get()
        level = self._safe_level(prefix, self.plot_level)
        if not prefix:
            messagebox.showwarning("Playback", "Choose a variable on the Field panel.")
            return None
        iters = self._iters_for(prefix)
        if not iters:
            messagebox.showwarning("Playback", f"{prefix} has no snapshots.")
            return None
        if level is None:
            messagebox.showwarning("Playback", "Choose a level.")
            return None
        return prefix, level, iters

    def _draw_plot_at_iteration(self, iteration: int) -> None:
        prefix = self.plot_var.get()
        level = self._safe_level(prefix, self.plot_level)
        if not prefix or level is None:
            return
        field2d = io.read_level_slice(self.data_dir.get(), prefix, iteration, level)
        iters = self._iters_for(prefix)
        anchor = self._playback_anchor_iter if self._playing else None
        clim_key = self._playback_clim_key_for(prefix, iters, level, anchor)
        use_locked = (
            self.playback_lock_scale.get()
            and self._playback_clim_key == clim_key
        )
        vmin_in = self._playback_vmin if use_locked else self._parse_float(self.plot_vmin.get())
        vmax_in = self._playback_vmax if use_locked else self._parse_float(self.plot_vmax.get())
        self._clear_figure()
        plotting.draw_slice_on_ax(
            self.ax, field2d, self.data_dir.get(),
            title=f"{prefix} · iter {iteration}{self._level_phrase(prefix, level)}",
            cmap=self.plot_cmap.get(),
            vmin=vmin_in,
            vmax=vmax_in,
            clear=False, colorbar_label=prefix,
            **self._grid_plot_kwargs(),
        )
        self._update_field_stats(field2d)
        iter_slider = self._plot_card.plot_iter_slider
        if self._playing:
            iter_slider.set_display_only(iteration)
        else:
            iter_slider.set_value(iteration)
        self._set_view(
            f"{prefix}  ·  iteration {iteration}{self._level_phrase(prefix, level)}",
            plotting.format_cmap_limits_meta(
                self.plot_cmap.get(), field2d, vmin_in, vmax_in,
            ),
        )
        self.canvas.draw()

    def toggle_playback(self) -> None:
        if self._playing:
            self._pause_playback()
        else:
            self._start_playback()

    def _start_playback(self) -> None:
        settings = self._playback_settings()
        if not settings:
            return
        prefix, level, iters = settings
        self._playback_iters = iters
        if self._has_iteration(prefix, self.plot_iter.get()):
            self._playback_index = self._iter_map[prefix].index(int(self.plot_iter.get()))
        else:
            self._playback_index = 0
        self._playback_anchor_iter = iters[self._playback_index]
        self._ensure_playback_clim(prefix, iters, level, anchor_iteration=self._playback_anchor_iter)
        self._playing = True
        self._plot_card.plot_iter_slider.set_playback_mode(True)
        if self._play_btn:
            self._play_btn.configure(text="Pause")
        self._log(f"Playing {prefix} · {len(iters)} frames")
        self._playback_tick()

    def _pause_playback(self) -> None:
        self._playing = False
        if self._playback_job:
            with contextlib.suppress(tk.TclError):
                self.after_cancel(self._playback_job)
            self._playback_job = None
        if hasattr(self, "_plot_card"):
            slider = self._plot_card.plot_iter_slider
            slider.set_playback_mode(False)
            slider.sync_slider()
        if self._play_btn:
            self._play_btn.configure(text="Play")
        self._playback_anchor_iter = None

    def stop_playback(self) -> None:
        self._pause_playback()
        if self._playback_iters:
            self._playback_index = 0
            self._draw_plot_at_iteration(self._playback_iters[0])

    def _playback_tick(self) -> None:
        if self._closing or not self._playing or not self._playback_iters:
            return
        if self._playback_index >= len(self._playback_iters):
            if self.playback_loop.get():
                self._playback_index = 0
            else:
                self._pause_playback()
                return
        frame_idx = self._playback_index
        self._draw_plot_at_iteration(self._playback_iters[frame_idx])
        self._playback_index += 1
        delay = self._playback_delay_ms(frame_idx)
        if not self._closing:
            self._playback_job = self.after(delay, self._playback_tick)

    def step_playback_frame(self, delta: int) -> None:
        settings = self._playback_settings()
        if not settings:
            return
        prefix, level, iters = settings
        self._pause_playback()
        self._playback_iters = iters
        if self._has_iteration(prefix, self.plot_iter.get()):
            idx = self._iter_map[prefix].index(int(self.plot_iter.get()))
        else:
            idx = 0
        idx = max(0, min(len(iters) - 1, idx + delta))
        self._playback_index = idx
        self._draw_plot_at_iteration(iters[idx])

    def jump_playback_frame(self, to_end: int) -> None:
        settings = self._playback_settings()
        if not settings:
            return
        prefix, level, iters = settings
        self._pause_playback()
        self._playback_iters = iters
        idx = 0 if to_end == 0 else len(iters) - 1
        self._draw_plot_at_iteration(iters[idx])

    def show_gif_dialog(self) -> None:
        settings = self._playback_settings()
        if not settings:
            return
        prefix, level, iters = settings

        dlg = ctk.CTkToplevel(self)
        dlg.title("Export GIF")
        dlg.geometry("480x520")
        dlg.minsize(440, 460)
        dlg.configure(fg_color=C["surface"])
        dlg.transient(self)
        dlg.grab_set()

        root = ctk.CTkFrame(dlg, fg_color="transparent")
        root.pack(fill=tk.BOTH, expand=True)
        root.grid_columnconfigure(0, weight=1)
        root.grid_rowconfigure(0, weight=1)

        scroll = ctk.CTkScrollableFrame(
            root,
            fg_color="transparent",
            scrollbar_button_color=C["border"],
            scrollbar_button_hover_color=C["accent_soft"],
        )
        scroll.grid(row=0, column=0, sticky="nsew", padx=20, pady=(20, 8))

        ctk.CTkLabel(scroll, text="Export animation", font=G.FONTS["heading"], text_color=C["text"]).pack(
            anchor="w"
        )
        ctk.CTkLabel(
            scroll,
            text="Uses Field panel: variable, level, colormap, and colour limits.",
            font=G.FONTS["caption"],
            text_color=C["muted"],
            wraplength=400,
            justify="left",
        ).pack(anchor="w", pady=(4, 12))

        summary = G.card_frame(scroll)
        summary.pack(fill=tk.X, pady=(0, 14))
        inner = ctk.CTkFrame(summary, fg_color="transparent")
        inner.pack(fill=tk.X, padx=14, pady=12)
        for label, value in (
            ("Variable", prefix),
            ("Level", str(level)),
            ("Frames", str(len(iters))),
            ("Colormap", self.plot_cmap.get()),
        ):
            row = ctk.CTkFrame(inner, fg_color="transparent")
            row.pack(fill=tk.X, pady=3)
            ctk.CTkLabel(
                row, text=label, width=100, anchor="w", font=G.FONTS["caption"], text_color=C["muted"]
            ).pack(side=tk.LEFT)
            ctk.CTkLabel(row, text=value, anchor="w", font=G.FONTS["label"], text_color=C["text"]).pack(
                side=tk.LEFT, fill=tk.X, expand=True
            )

        G.section_label(scroll, "Animation").pack(anchor="w", pady=(4, 0))
        fps_row = ctk.CTkFrame(scroll, fg_color="transparent")
        fps_row.pack(fill=tk.X, pady=(6, 10))
        ctk.CTkLabel(fps_row, text="Frames per second", font=G.FONTS["label"], text_color=C["text"]).pack(
            side=tk.LEFT
        )
        ctk.CTkEntry(
            fps_row,
            textvariable=self.playback_fps,
            width=72,
            height=G.CTK["entry_height"],
            font=G.FONTS["label"],
            fg_color=C["card"],
            border_color=C["border_strong"],
            border_width=G.CTK["border_width"],
        ).pack(side=tk.RIGHT)

        ctk.CTkCheckBox(
            scroll,
            text="Lock colour scale (playback: from start frame; GIF: all frames)",
            variable=self.playback_lock_scale,
            font=G.FONTS["label"],
            text_color=C["text"],
        ).pack(anchor="w", pady=(0, 14))

        G.section_label(scroll, "Colour limits").pack(anchor="w")
        ctk.CTkLabel(
            scroll,
            text="Leave blank to auto-scale (from Field panel when unlocked).",
            font=G.FONTS["caption"],
            text_color=C["muted"],
            wraplength=400,
            justify="left",
        ).pack(anchor="w", pady=(4, 8))
        lim_row = ctk.CTkFrame(scroll, fg_color="transparent")
        lim_row.pack(fill=tk.X, pady=(0, 8))
        gif_vmin = tk.StringVar(value=self.plot_vmin.get())
        gif_vmax = tk.StringVar(value=self.plot_vmax.get())
        G.entry_field(lim_row, gif_vmin, width=110).pack(side=tk.LEFT, padx=(0, 8))
        ctk.CTkLabel(lim_row, text="to", font=G.FONTS["caption"], text_color=C["muted"]).pack(side=tk.LEFT, padx=4)
        G.entry_field(lim_row, gif_vmax, width=110).pack(side=tk.LEFT)

        btn_row = ctk.CTkFrame(root, fg_color="transparent")
        btn_row.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 20))

        def do_export() -> None:
            path = filedialog.asksaveasfilename(
                parent=dlg,
                defaultextension=".gif",
                filetypes=[("GIF", "*.gif")],
                initialfile=f"{prefix}_L{level}.gif",
            )
            if not path:
                return
            dlg.destroy()
            self._export_gif(
                path,
                prefix,
                level,
                iters,
                vmin=self._parse_float(gif_vmin.get()),
                vmax=self._parse_float(gif_vmax.get()),
            )

        G.accent_button(btn_row, "Export…", do_export).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        G.ghost_button(btn_row, "Cancel", dlg.destroy, width=100).pack(side=tk.LEFT)

        dlg.bind("<Escape>", lambda _e: dlg.destroy())
        dlg.update_idletasks()
        px = self.winfo_rootx() + max((self.winfo_width() - dlg.winfo_width()) // 2, 0)
        py = self.winfo_rooty() + max((self.winfo_height() - dlg.winfo_height()) // 2, 0)
        dlg.geometry(f"+{px}+{py}")
        dlg.focus_force()

    def _export_gif(
        self,
        path: str,
        prefix: str,
        level: int,
        iters: list[int],
        *,
        vmin: float | None,
        vmax: float | None,
    ) -> None:
        self._pause_playback()
        data_dir, cmap = self.data_dir.get(), self.plot_cmap.get()
        fps, lock = float(self.playback_fps.get()), self.playback_lock_scale.get()

        def work() -> None:
            try:
                animation.save_playback_gif(
                    data_dir,
                    prefix,
                    iters,
                    level,
                    path,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    lock_scale=lock,
                    fps=fps,
                    coord_mode=self.grid_coord_mode.get(),
                    overlay_grid=self.grid_overlay.get(),
                    progress=lambda d, t: self._call_ui(lambda: self.status.set(f"GIF {d}/{t}")),
                )
                self._call_ui(lambda: self._log(f"Saved {path}"))
            except Exception as exc:
                self._call_ui(lambda: messagebox.showerror("GIF failed", str(exc)))

        self.status.set("Writing GIF…")
        threading.Thread(target=work, daemon=True).start()

    def save_gif(self) -> None:
        """Open the GIF export dialog (kept for any external callers)."""
        self.show_gif_dialog()

    def _call_ui(self, callback) -> None:
        """Run a callback on the Tk main thread if the window is still open."""
        if self._closing:
            return
        try:
            self.after(0, callback)
        except tk.TclError:
            pass

    def _cancel_all_after(self) -> None:
        """Cancel every pending Tk after callback (including CTk internals)."""
        try:
            for after_id in self.tk.call("after", "info"):
                with contextlib.suppress(tk.TclError):
                    self.after_cancel(after_id)
        except tk.TclError:
            pass

    def _on_close(self) -> None:
        if self._closing:
            return
        self._closing = True
        self._pause_playback()
        if self._auto_plot_job:
            with contextlib.suppress(tk.TclError):
                self.after_cancel(self._auto_plot_job)
            self._auto_plot_job = None
        try:
            self.withdraw()
        except tk.TclError:
            pass
        self._cancel_all_after()
        with contextlib.suppress(Exception):
            plt.close(self.fig)
        with contextlib.suppress(tk.TclError):
            self.quit()
        with contextlib.suppress(tk.TclError):
            self.destroy()

    # ---------------------------------------------------------------- actions
    def _clear_figure(self) -> None:
        self.fig.clf()
        self.ax = self.fig.add_subplot(111)

    def plot_snapshot(self) -> None:
        if self._closing:
            return
        self._pause_playback()
        prefix = self.plot_var.get()
        iteration = self._parse_int(self.plot_iter.get())
        level = self._safe_level(prefix, self.plot_level)
        if not prefix or iteration is None or level is None:
            return
        try:
            self._draw_plot_at_iteration(iteration)
            self._log(f"{prefix} · iter {iteration} · level {level}")
        except Exception as exc:
            messagebox.showerror("Plot failed", str(exc))

    def plot_diff(self) -> None:
        prefix = self.diff_var.get()
        later_dir = self.diff_later_dir.get()
        earlier_dir = self.diff_earlier_dir.get()
        later, earlier = self._parse_int(self.diff_later.get()), self._parse_int(self.diff_earlier.get())
        level = self._safe_level(prefix, self.diff_level, later_dir)
        if None in (later, earlier) or level is None:
            return
        try:
            diff, _ = ops.diff_slice(
                later_dir, prefix, later, earlier, level=level, data_dir_b=earlier_dir
            )
            diff2d = np.squeeze(diff)
            vmin_in = self._parse_float(self.diff_vmin.get())
            vmax_in = self._parse_float(self.diff_vmax.get())
            self._clear_figure()
            same_dir = os.path.abspath(later_dir) == os.path.abspath(earlier_dir)
            if same_dir:
                title = f"{prefix}({later}) − {prefix}({earlier})"
                meta_line = f"iter {later} − {earlier}  ·  level {level}"
            else:
                title = (
                    f"{prefix}({later}@{self._short_dir(later_dir)}) − "
                    f"{prefix}({earlier}@{self._short_dir(earlier_dir)})"
                )
                meta_line = f"{self._short_dir(later_dir)}@{later} − {self._short_dir(earlier_dir)}@{earlier}  ·  level {level}"
            plotting.plot_array(
                diff, later_dir,
                title=title,
                level=None, cmap=self.diff_cmap.get(),
                vmin=vmin_in, vmax=vmax_in,
                symmetric=True, ax=self.ax, show=False,
                **self._grid_plot_kwargs(),
            )
            self.canvas.draw()
            self._update_field_stats(diff2d)
            self._set_view(
                f"Δ {prefix}",
                f"{meta_line}  ·  {plotting.format_cmap_limits_meta(self.diff_cmap.get(), diff2d, vmin_in, vmax_in, symmetric=True)}",
            )
            self._log(f"Diff {prefix}")
        except Exception as exc:
            messagebox.showerror("Diff failed", str(exc))

    def save_diff_field(self) -> None:
        prefix = self.diff_var.get()
        later_dir = self.diff_later_dir.get()
        earlier_dir = self.diff_earlier_dir.get()
        later, earlier = self._parse_int(self.diff_later.get()), self._parse_int(self.diff_earlier.get())
        if None in (later, earlier):
            return
        path = filedialog.asksaveasfilename(initialfile=f"Diff_{prefix}_{later}_{earlier}")
        if not path:
            return
        try:
            diff, _ = ops.diff_fields(
                later_dir, prefix, later, earlier, data_dir_b=earlier_dir
            )
            io.write_field(path, diff, iteration=later)
            self._log(f"Saved diff → {path}")
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))

    def plot_dod(self) -> None:
        var_a, var_b = self.dod_a.get(), self.dod_b.get()
        t1, t2 = self._parse_int(self.dod_t1.get()), self._parse_int(self.dod_t2.get())
        level = self._safe_level(var_a, self.dod_level)
        if not var_a or not var_b or None in (t1, t2) or level is None:
            return
        try:
            result, meta = ops.difference_of_differences(
                self.data_dir.get(), var_a, var_b, t1, t2, levels=[level], progress=False,
            )
            result2d = np.squeeze(np.asarray(result))
            vmin_in = self._parse_float(self.dod_vmin.get())
            vmax_in = self._parse_float(self.dod_vmax.get())
            self._clear_figure()
            plotting.plot_array(
                result, self.data_dir.get(), title="DiD", level=0, cmap=self.dod_cmap.get(),
                vmin=vmin_in, vmax=vmax_in,
                symmetric=True, ax=self.ax, show=False,
                **self._grid_plot_kwargs(),
            )
            self.canvas.draw()
            summary = ops.stats(np.asarray(result))
            self._update_field_stats(result2d)
            self._set_view(
                f"DiD  ·  level {level}",
                (
                    f"({var_b}−{var_a})@t1 − ({var_b}−{var_a})@t2   ·   mean {summary['mean']:.4g}"
                    f"  ·  {plotting.format_cmap_limits_meta(self.dod_cmap.get(), result2d, vmin_in, vmax_in, symmetric=True)}"
                ),
            )
            self._close_mmap(result, meta)
        except Exception as exc:
            messagebox.showerror("DiD failed", str(exc))

    def show_dod_stats(self) -> None:
        t1, t2 = self._parse_int(self.dod_t1.get()), self._parse_int(self.dod_t2.get())
        if None in (t1, t2):
            return
        try:
            summary = ops.streaming_stats(self.data_dir.get(), self.dod_a.get(), self.dod_b.get(), t1, t2)
            text = "\n".join(f"{k:>8s}  {v:.6g}" for k, v in summary.items())
            messagebox.showinfo("DiD statistics", text)
        except Exception as exc:
            messagebox.showerror("Stats failed", str(exc))

    def save_dod_field(self) -> None:
        var_a, var_b = self.dod_a.get(), self.dod_b.get()
        t1, t2 = self._parse_int(self.dod_t1.get()), self._parse_int(self.dod_t2.get())
        if None in (t1, t2):
            return
        path = filedialog.asksaveasfilename(initialfile=f"DiD_{var_a}_{var_b}_{t1}_{t2}")
        if not path:
            return
        try:
            result, meta = ops.difference_of_differences(
                self.data_dir.get(), var_a, var_b, t1, t2, progress=True,
            )
            io.write_field(path, result, iteration=t1)
            self._log(f"Saved DiD → {path}")
            self._close_mmap(result, meta)
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))

    def save_png(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if path:
            self.fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=C["figure_bg"])
            self._log(f"Saved {path}")

    @staticmethod
    def _parse_float(text: str) -> float | None:
        text = text.strip()
        return float(text) if text else None

    @staticmethod
    def _parse_int(text: str) -> int | None:
        text = text.strip()
        return int(text) if text else None

    @staticmethod
    def _close_mmap(result, meta) -> None:
        if hasattr(result, "base") and hasattr(result.base, "close"):
            result.base.close()
        mmap_path = meta.get("mmap_path")
        if mmap_path and os.path.exists(mmap_path):
            os.unlink(mmap_path)


def launch_gui(initial_dir: str = ".") -> None:
    MdsViewApp(initial_dir=initial_dir).mainloop()
