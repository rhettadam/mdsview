"""Visual theme for the mdsview GUI (CustomTkinter + matplotlib)."""

from __future__ import annotations

import sys
import tkinter as tk
import tkinter.font as tkfont
from pathlib import Path
from tkinter import ttk

import customtkinter as ctk
from PIL import Image

APP_NAME = "mdsview"
LOGO_PATH = Path(__file__).resolve().parent / "assets" / "logo.png"
LOGO_HEIGHT = 44

# Populated by resolve_fonts() at startup
UI_FONT = "sans-serif"
MONO_FONT = "monospace"

# Dark neutral shell — cool grays, restrained blue accent
COLORS = {
    "bg": "#181a1f",
    "surface": "#1e2128",
    "card": "#252830",
    "nav": "#1a1d24",
    "nav_active": "#2d323c",
    "border": "#363b47",
    "border_strong": "#454b5a",
    "text": "#e4e6eb",
    "muted": "#8b92a0",
    "accent": "#4c8bf5",
    "accent_hover": "#6ba1ff",
    "accent_soft": "#2a3548",
    "header_text": "#e4e6eb",
    "list_select": "#3d5a99",
    "figure_bg": "#ffffff",
    "axes_bg": "#2a2e36",
    "stat_bg": "#1e2128",
    "status_bg": "#14161a",
}

FONTS = {
    "brand": (UI_FONT, 26, "bold"),
    "heading": (UI_FONT, 14, "bold"),
    "label": (UI_FONT, 12),
    "caption": (UI_FONT, 11),
    "small": (UI_FONT, 10),
    "mono": (MONO_FONT, 11),
    "stat": (MONO_FONT, 12),
    "nav": (UI_FONT, 11),
}

CTK = {
    "corner_radius": 8,
    "border_width": 1,
    "button_height": 34,
    "entry_height": 34,
}


def init_app(root: tk.Misc | None = None) -> None:
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    resolve_fonts(root)


def resolve_fonts(root: tk.Misc | None = None) -> None:
    """Pick UI fonts available on this system (Linux-friendly fallbacks)."""
    global UI_FONT, MONO_FONT, FONTS

    if root is not None:
        families = set(tkfont.families(root))
    else:
        families = set()

    if sys.platform.startswith("win"):
        ui_candidates = ("Segoe UI", "DejaVu Sans", "Arial", "TkDefaultFont")
        mono_candidates = ("Consolas", "DejaVu Sans Mono", "Courier New", "TkFixedFont")
    elif sys.platform == "darwin":
        ui_candidates = ("SF Pro Text", "Helvetica Neue", "Helvetica", "Arial")
        mono_candidates = ("Menlo", "Monaco", "Courier New")
    else:
        ui_candidates = ("DejaVu Sans", "Liberation Sans", "Ubuntu", "Noto Sans", "Arial")
        mono_candidates = ("DejaVu Sans Mono", "Liberation Mono", "Ubuntu Mono", "Courier New")

    UI_FONT = next((name for name in ui_candidates if name in families), "sans-serif")
    MONO_FONT = next((name for name in mono_candidates if name in families), "monospace")
    FONTS = {
        "brand": (UI_FONT, 26, "bold"),
        "heading": (UI_FONT, 14, "bold"),
        "label": (UI_FONT, 12),
        "caption": (UI_FONT, 11),
        "small": (UI_FONT, 10),
        "mono": (MONO_FONT, 11),
        "stat": (MONO_FONT, 12),
        "nav": (UI_FONT, 11),
    }


def load_logo_image(*, height: int = LOGO_HEIGHT) -> ctk.CTkImage | None:
    """Return a CTkImage for the app logo, or None if the file is missing."""
    if not LOGO_PATH.is_file():
        return None
    with Image.open(LOGO_PATH) as img:
        source = img.copy()
    aspect = source.width / max(source.height, 1)
    size = (max(1, int(height * aspect)), height)
    return ctk.CTkImage(light_image=source, dark_image=source, size=size)


def configure_matplotlib() -> None:
    """Leave matplotlib defaults; plots match CLI and standard output."""


def apply_ttk_combobox_style(root: tk.Misc) -> ttk.Style:
    style = ttk.Style(root)
    style.theme_use("clam")
    c = COLORS
    style.configure(
        "Mds.TCombobox",
        fieldbackground=c["card"],
        background=c["card"],
        foreground=c["text"],
        arrowcolor=c["accent"],
        bordercolor=c["border_strong"],
        font=FONTS["label"],
    )
    return style


def card_frame(parent, **kwargs) -> ctk.CTkFrame:
    return ctk.CTkFrame(
        parent,
        fg_color=COLORS["card"],
        border_color=COLORS["border"],
        border_width=CTK["border_width"],
        corner_radius=CTK["corner_radius"],
        **kwargs,
    )


def section_label(parent, text: str) -> ctk.CTkLabel:
    return ctk.CTkLabel(
        parent,
        text=text,
        font=FONTS["small"],
        text_color=COLORS["muted"],
        anchor="w",
    )


def body_label(parent, text: str, **kwargs) -> ctk.CTkLabel:
    return ctk.CTkLabel(parent, text=text, font=FONTS["label"], text_color=COLORS["text"], **kwargs)


def accent_button(parent, text: str, command, **kwargs) -> ctk.CTkButton:
    return ctk.CTkButton(
        parent,
        text=text,
        command=command,
        font=FONTS["label"],
        fg_color=COLORS["accent"],
        hover_color=COLORS["accent_hover"],
        text_color="#ffffff",
        border_width=0,
        corner_radius=CTK["corner_radius"],
        height=CTK["button_height"],
        **kwargs,
    )


def ghost_button(parent, text: str, command, **kwargs) -> ctk.CTkButton:
    return ctk.CTkButton(
        parent,
        text=text,
        command=command,
        font=FONTS["caption"],
        fg_color=COLORS["surface"],
        hover_color=COLORS["accent_soft"],
        text_color=COLORS["text"],
        border_color=COLORS["border_strong"],
        border_width=CTK["border_width"],
        corner_radius=CTK["corner_radius"],
        height=32,
        **kwargs,
    )


def nav_button(parent, text: str, command) -> ctk.CTkButton:
    return ctk.CTkButton(
        parent,
        text=text,
        command=command,
        width=76,
        height=34,
        font=FONTS["nav"],
        fg_color="transparent",
        hover_color=COLORS["accent_soft"],
        text_color=COLORS["muted"],
        corner_radius=8,
    )


def option_menu(parent, variable: tk.StringVar, values: list[str], command=None) -> ctk.CTkOptionMenu:
    return ctk.CTkOptionMenu(
        parent,
        variable=variable,
        values=values or [""],
        command=command,
        font=FONTS["label"],
        fg_color=COLORS["surface"],
        button_color=COLORS["border_strong"],
        button_hover_color=COLORS["accent"],
        dropdown_fg_color=COLORS["card"],
        dropdown_hover_color=COLORS["accent_soft"],
        text_color=COLORS["text"],
        dropdown_font=FONTS["label"],
        corner_radius=6,
        height=CTK["entry_height"],
    )


def entry_field(parent, variable: tk.StringVar, width: int = 80) -> ctk.CTkEntry:
    return ctk.CTkEntry(
        parent,
        textvariable=variable,
        font=FONTS["label"],
        fg_color=COLORS["surface"],
        border_color=COLORS["border_strong"],
        border_width=CTK["border_width"],
        text_color=COLORS["text"],
        corner_radius=6,
        height=CTK["entry_height"],
        width=width,
    )


def style_listbox(listbox: tk.Listbox) -> None:
    c = COLORS
    listbox.configure(
        bg=c["surface"],
        fg=c["text"],
        selectbackground=c["list_select"],
        selectforeground="#ffffff",
        highlightthickness=CTK["border_width"],
        highlightbackground=c["border_strong"],
        highlightcolor=c["accent"],
        borderwidth=0,
        activestyle="none",
        font=FONTS["mono"],
    )
