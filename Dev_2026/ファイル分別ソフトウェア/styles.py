APP_STYLES = """
QWidget {
    background-color: #faf9f6;
    color: #333333;
}
QPushButton {
    border: none;
    border-radius: 6px;
    font-weight: bold;
}
QPushButton:hover {
    opacity: 0.9;
}
QPushButton:pressed {
    background-color: #4488cc;
}
QGroupBox {
    font-weight: bold;
    font-size: 15px;
    color: #2a688d;
    border: 2px solid #e0e0e0;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 8px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0px 8px 0px 0px;
    color: #4da6ff;
}
QLineEdit {
    border: 1px solid #cccccc;
    border-radius: 4px;
    padding: 6px;
    background-color: #ffffff;
}
QComboBox {
    border: 1px solid #cccccc;
    border-radius: 4px;
    padding: 6px;
    background-color: #ffffff;
}
QComboBox::drop-down {
    border: none;
    padding-left: 4px;
}
QTableView {
    background-color: #ffffff;
    border: 1px solid #e0e0e0;
    border-radius: 4px;
    gridline-color: #f0f0f0;
    alternate-background-color: #f9f9f9;
}
QHeaderView::horizontal {
    background-color: #e8e8e8;
    color: #222222;
    padding: 8px;
    border-bottom: 1px solid #d0d0d0;
    font-weight: bold;
}
QScrollBar:horizontal {
    height: 10px;
    background-color: #f0f0f0;
    border-radius: 5px;
    margin: 2px;
}
QScrollBar::handle:horizontal {
    background-color: #c0c0c0;
    border-radius: 5px;
    min-width: 20px;
}
QScrollBar::add-line:horizontal {
    background-color: #a0a0a0;
    border-radius: 5px;
    width: 20px;
}
QScrollBar::sub-line:horizontal {
    background-color: #a0a0a0;
    border-radius: 5px;
    width: 20px;
}
QScrollBar:vertical {
    width: 10px;
    background-color: #f0f0f0;
    border-radius: 5px;
    margin: 2px;
}
QScrollBar::handle:vertical {
    background-color: #c0c0c0;
    border-radius: 5px;
    min-height: 20px;
}
QScrollBar::add-line:vertical {
    background-color: #a0a0a0;
    border-radius: 5px;
    height: 20px;
}
QScrollBar::sub-line:vertical {
    background-color: #a0a0a0;
    border-radius: 5px;
    height: 20px;
}
QLabel {
    background-color: transparent;
}
"""

COLOR_PALETTE = {
    "py": {"bg": "#e8f5e9", "fg": "#2e7d32"},
    "txt": {"bg": "#f5f5f5", "fg": "#333333"},
    "md": {"bg": "#e3f2fd", "fg": "#1565c0"},
    "json": {"bg": "#fffde7", "fg": "#fbc02d"},
    "xml": {"bg": "#ffffff", "fg": "#333333"},
    "html": {"bg": "#fce4ec", "fg": "#c2185b"},
    "js": {"bg": "#f3e5f5", "fg": "#7b1fa2"},
    "css": {"bg": "#e0f7fa", "fg": "#0097a7"},
    "java": {"bg": "#f3e5f5", "fg": "#7b1fa2"},
    "png": {"bg": "#e8f5e9", "fg": "#2e7d32"},
    "jpg": {"bg": "#e8f5e9", "fg": "#2e7d32"},
    "pdf": {"bg": "#fce4ec", "fg": "#c2185b"},
    "zip": {"bg": "#fffde7", "fg": "#fbc02d"},
}