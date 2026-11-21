"""Small shared building blocks for the AI / MODEL and MATCH screens: panels, labels, key/value grids, tables, one style sheet.

Same look as the CONNECTIONS tab. No logic and no backend access here.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QAbstractItemView, QFrame, QGridLayout, QHeaderView, QLabel, QTableWidget, QTableWidgetItem, QVBoxLayout,
                               QWidget)

LEVEL_COLOURS = {"ok": "#55c889", "warn": "#e8b85b", "bad": "#ef6b73", "off": "#9aa5b1", "plain": "#d9dee5", "blocked": "#b48be8"}

STYLE = """
QLabel { background: transparent; color: #d9dee5; }
QFrame#panel { background: #171e27; border: 1px solid #2b3542; border-radius: 5px; }
QLabel#panelTitle { color: #9eabb8; font-size: 10pt; font-weight: 700; letter-spacing: 1px; }
QLabel#screenTitle { color: #d9dee5; font-size: 15pt; font-weight: 800; letter-spacing: 1px; }
QLabel#hint { color: #9eabb8; }
QLabel#rowName { color: #9eabb8; }
QLabel#headline { font-size: 13pt; font-weight: 800; }
QLabel#timer { font-size: 30pt; font-weight: 800; font-family: Consolas; }
QLabel#banner { border-radius: 5px; padding: 8px 12px; font-size: 11pt; font-weight: 700; }
QLabel#chip { border-radius: 10px; padding: 3px 12px; font-weight: 800; color: #0d1218; }
QTableWidget { background: #0d1218; alternate-background-color: #121922; gridline-color: #2b3542; color: #d9dee5;
               border: 1px solid #2b3542; }
QHeaderView::section { background: #171e27; color: #9eabb8; border: none; border-bottom: 1px solid #2b3542; padding: 4px 6px; font-weight: 700; }
QLineEdit, QComboBox, QPlainTextEdit, QListWidget { background: #0d1218; border: 1px solid #3b4a5d; border-radius: 4px; padding: 5px 6px; color: #d9dee5; }
QPlainTextEdit { font-family: Consolas; }
QScrollArea { border: none; background: transparent; }
QWidget#scrollBody { background: transparent; }
QPushButton { padding: 6px 14px; }
QPushButton:disabled { color: #6b7684; }
QPushButton#modeButton { font-size: 13pt; font-weight: 800; padding: 14px 22px; border: 2px solid #3b4a5d; border-radius: 6px; }
QPushButton#modeButton:checked { background: #2f4a66; border-color: #7fb2e5; color: white; }
QPushButton#startButton { font-size: 14pt; font-weight: 800; padding: 14px 26px; background: #2c6b48; color: white; border-radius: 6px; }
QPushButton#startButton:disabled { background: #232b34; color: #6b7684; }
QPushButton#stopButton { font-size: 14pt; font-weight: 800; padding: 14px 26px; background: #a8472f; color: white; border-radius: 6px; }
QPushButton#emergencyButton { font-size: 14pt; font-weight: 900; padding: 14px 26px; background: #c4222e; color: white; border-radius: 6px; }
"""


def label(text: str = "", name: str | None = None, selectable: bool = True) -> QLabel:
    item = QLabel(text)
    item.setTextFormat(Qt.TextFormat.PlainText)
    item.setWordWrap(True)
    if name:
        item.setObjectName(name)
    if selectable:
        item.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return item


def set_text(item: QLabel, text: str, level: str | None = None) -> None:
    if item.text() != text:
        item.setText(text)
    if level is not None and item.property("level") != level:
        item.setProperty("level", level)
        item.setStyleSheet(f"color: {LEVEL_COLOURS.get(level, LEVEL_COLOURS['plain'])};")


def panel(title: str) -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame()
    frame.setObjectName("panel")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(12, 10, 12, 12)
    layout.setSpacing(6)
    heading = QLabel(title)
    heading.setObjectName("panelTitle")
    layout.addWidget(heading)
    return frame, layout


class KeyValueGrid(QWidget):
    """Rows of (name, value, level); the labels are reused so a refresh does not rebuild widgets."""

    def __init__(self) -> None:
        super().__init__()
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(14)
        self._grid.setColumnStretch(1, 1)
        self._names: list[QLabel] = []
        self._values: list[QLabel] = []

    def set_rows(self, rows: list[tuple[str, str, str]]) -> None:
        while len(self._names) < len(rows):
            row = len(self._names)
            name, value = label("", "rowName", selectable=False), label("")
            self._grid.addWidget(name, row, 0, Qt.AlignmentFlag.AlignTop)
            self._grid.addWidget(value, row, 1)
            self._names.append(name)
            self._values.append(value)
        for index, (name, value, level) in enumerate(rows):
            self._names[index].show()
            self._values[index].show()
            set_text(self._names[index], name)
            set_text(self._values[index], value, level)
        for index in range(len(rows), len(self._names)):
            self._names[index].hide()
            self._values[index].hide()

    def texts(self) -> list[tuple[str, str]]:
        return [(n.text(), v.text()) for n, v in zip(self._names, self._values) if not n.isHidden()]


def make_table(headers: list[str], stretch_last: bool = True) -> QTableWidget:
    table = QTableWidget(0, len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.verticalHeader().setVisible(False)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
    table.setAlternatingRowColors(True)
    table.verticalHeader().setDefaultSectionSize(24)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    table.horizontalHeader().setStretchLastSection(stretch_last)
    return table


def fill_table(table: QTableWidget, rows: list[tuple[list[str], str]]) -> None:
    """rows: ([cell texts], level)."""
    table.setRowCount(len(rows))
    for r, (cells, level) in enumerate(rows):
        for c, text in enumerate(cells):
            item = table.item(r, c)
            if item is None:
                item = QTableWidgetItem()
                table.setItem(r, c, item)
            if item.text() != text:
                item.setText(text)
            item.setForeground(QColor(LEVEL_COLOURS.get(level, LEVEL_COLOURS["plain"])))


def table_texts(table: QTableWidget) -> list[list[str]]:
    return [[(table.item(r, c).text() if table.item(r, c) else "") for c in range(table.columnCount())] for r in range(table.rowCount())]
