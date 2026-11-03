import os
from pathlib import Path
from typing import List, Optional, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

_IGNORE_KEYWORDS = (
    "uninstall",
    "remove",
    "readme",
    "help",
    "manual",
    "documentation",
    "website",
    "release notes",
)


def discover_windows_programs() -> List[Tuple[str, str]]:
    """Scan Windows Start Menu directories for recognized installed programs (.lnk / .appref-ms)."""
    roots: List[Path] = []
    appdata = os.environ.get("APPDATA")
    programdata = os.environ.get("PROGRAMDATA")
    if appdata:
        roots.append(Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs")
    if programdata:
        roots.append(Path(programdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs")

    found: dict[str, str] = {}
    for root in roots:
        if not root.exists():
            continue
        for ext in ("*.lnk", "*.appref-ms"):
            for path in root.rglob(ext):
                name = path.stem.strip()
                if not name:
                    continue
                lower_name = name.lower()
                if any(kw in lower_name for kw in _IGNORE_KEYWORDS):
                    continue
                if name not in found:
                    found[name] = str(path)

    return sorted(found.items(), key=lambda item: item[0].lower())


def format_exec_label(path_str: str) -> str:
    p = Path(path_str)
    name = p.stem if p.stem else p.name
    return f"Run: {name}"


class ProgramPickerDialog(QDialog):
    """Dialog to choose an installed Windows Start Menu program or browse for any file/.exe."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select Program or File to Execute")
        self.resize(460, 480)
        self.selected_path: Optional[str] = None

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        info = QLabel(
            "Choose a program recognized by Windows from the menu below, "
            "or click 'Browse File / .exe...' to pick any file on disk:",
            self,
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        # Search filter
        self.search_input = QLineEdit(self)
        self.search_input.setPlaceholderText("Search installed Windows programs (e.g. Spotify, Steam)...")
        self.search_input.textChanged.connect(self._filter_list)
        layout.addWidget(self.search_input)

        # Program list
        self.list_widget = QListWidget(self)
        self.list_widget.itemDoubleClicked.connect(self._on_item_double_clicked)
        layout.addWidget(self.list_widget, stretch=1)

        self._programs = discover_windows_programs()
        self._populate_list("")

        # Buttons row
        btn_row = QHBoxLayout()
        self.browse_btn = QPushButton("Browse File / .exe...", self)
        self.browse_btn.clicked.connect(self._on_browse_clicked)
        btn_row.addWidget(self.browse_btn)

        btn_row.addStretch(1)

        self.cancel_btn = QPushButton("Cancel", self)
        self.cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(self.cancel_btn)

        self.select_btn = QPushButton("Select Program", self)
        self.select_btn.setDefault(True)
        self.select_btn.clicked.connect(self._on_select_clicked)
        btn_row.addWidget(self.select_btn)

        layout.addLayout(btn_row)

    def _populate_list(self, query: str) -> None:
        self.list_widget.clear()
        q = query.strip().lower()
        for name, full_path in self._programs:
            if q and q not in name.lower():
                continue
            item = QListWidgetItem(name)
            item.setData(Qt.ItemDataRole.UserRole, full_path)
            item.setToolTip(full_path)
            self.list_widget.addItem(item)
        if self.list_widget.count() > 0:
            self.list_widget.setCurrentRow(0)

    def _filter_list(self, text: str) -> None:
        self._populate_list(text)

    def _on_item_double_clicked(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.ItemDataRole.UserRole)
        if path:
            self.selected_path = str(path)
            self.accept()

    def _on_select_clicked(self) -> None:
        item = self.list_widget.currentItem()
        if item is not None:
            path = item.data(Qt.ItemDataRole.UserRole)
            if path:
                self.selected_path = str(path)
                self.accept()

    def _on_browse_clicked(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Executable or File to Launch",
            "",
            "All Files (*.*);;Programs (*.exe *.lnk *.bat *.cmd)",
        )
        if file_path:
            self.selected_path = file_path
            self.accept()
