import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, 
    QPushButton, QLineEdit, QTextEdit, QTableWidget, QTableWidgetItem,
    QHeaderView, QFileDialog, QMessageBox, QGroupBox, QComboBox
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QColor
import os
import json

# スタイルモジュールをインポート
from file_explorer.styles import APP_STYLES, COLOR_PALETTE

class FileExplorer:
    """ファイル解析クラス"""
    
    def __init__(self):
        self.files = []
        self.groups = {}
        self.file_extensions = []
        self.total_files = 0
        self.total_size = 0
        self.folder_path = ""
    
    def select_folder(self):
        """フォルダ選択ダイアログを表示"""
        folder_path = QFileDialog.getExistingDirectory(
            None, "ファイル選択フォルダ",
            "Select a folder to analyze"
        )
        if folder_path:
            self.folder_path = folder_path
            self.load_files()
            return self.folder_path
        return None
    
    def load_files(self):
        """指定フォルダのファイルを走査"""
        self.files = []
        self.groups = {}
        self.file_extensions = []
        self.total_files = 0
        self.total_size = 0
        
        if not os.path.exists(self.folder_path):
            return
        
        for file_info in os.listdir(self.folder_path):
            file_path = os.path.join(self.folder_path, file_info)
            if os.path.isfile(file_path):
                self.total_files += 1
                self.add_file_to_list(file_path)
    
    def add_file_to_list(self, file_path):
        """ファイル情報を追加し、グループ化する"""
        file_size = os.path.getsize(file_path)
        extension = os.path.splitext(file_path)[1].lower()
        ext_name = os.path.splitext(file_path)[1][1:].lower() if extension else "no extension"
        
        file_entry = {
            "path": file_path,
            "name": os.path.basename(file_path),
            "extension": extension,
            "ext_name": ext_name,
            "size": file_size,
            "size_display": self.format_size(file_size)
        }
        
        self.files.append(file_entry)
        self.total_size += file_size
        
        if extension not in self.groups:
            self.groups[extension] = []
        self.groups[extension].append(file_entry)
    
    def format_size(self, bytes_size):
        """バイト数を人間読みにフォーマット"""
        if bytes_size < 1024:
            return str(bytes_size) + " B"
        elif bytes_size < 1024 * 1024:
            return f"{bytes_size / 1024:.2f} KB"
        elif bytes_size < 1024 * 1024 * 1024:
            return f"{bytes_size / (1024 * 1024):.2f} MB"
        else:
            return f"{bytes_size / (1024 * 1024 * 1024):.2f} GB"
    
    def get_extension_groups(self):
        """拡張子ごとのグループとファイル数を返す"""
        groups = {}
        for ext, files in self.groups.items():
            groups[ext] = {
                "count": len(files),
                "files": files,
                "total_size": sum(f["size"] for f in files),
                "total_size_display": self.format_size(sum(f["size"] for f in files))
            }
        return groups
    
    def get_all_extensions(self):
        """全ての拡張子を返す"""
        return self.file_extensions

class FileExplorerUI(QMainWindow):
    """メイン UI クラス"""
    
    def __init__(self):
        super().__init__()
        self.file_analyzer = FileExplorer()
        self.init_ui()
    
    def init_ui(self):
        """UI の初期化"""
        self.setWindowTitle("ファイル・エキスplorer")
        self.setMinimumWidth(900)
        self.setMinimumHeight(550)
        self.setStyleSheet(APP_STYLES)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # タイトル
        title_label = QLabel("📁 ファイル・エキスplorer")
        title_label.setFont(QFont("Hiragino Sans", 18, QFont.Bold))
        title_label.setStyleSheet("color: #2a688d;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # 1. 操作パネル
        control_group = QGroupBox("操作パネル")
        control_layout = QVBoxLayout(control_group)
        
        self.folder_label = QLabel("フォルダを選択して分析してください")
        self.folder_label.setStyleSheet("color: #666666; padding: 10px;")
        control_layout.addWidget(self.folder_label)
        
        self.folder_btn = QPushButton("📁 フォルダを選択")
        self.folder_btn.setFixedSize(200, 35)
        self.folder_btn.setStyleSheet("background-color: #4da6ff; color: white; border-radius: 6px; font-weight: bold;")
        self.folder_btn.clicked.connect(self.on_folder_selected)
        control_layout.addWidget(self.folder_btn)
        
        control_layout.addStretch()
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)
        
        # 2. 検索パネル
        search_group = QGroupBox("検索")
        search_layout = QVBoxLayout(search_group)
        
        self.search_input = QLineEdit("")
        self.search_input.setPlaceholderText("検索キーワード（ファイル名または拡張子）")
        self.search_input.setFixedWidth(250)
        self.search_input.setFixedHeight(28)
        self.search_input.setStyleSheet("border: 1px solid #cccccc; border-radius: 4px; padding: 6px; background-color: #fff;")
        
        self.search_btn = QPushButton("🔍 検索")
        self.search_btn.setFixedSize(100, 30)
        self.search_btn.setStyleSheet("background-color: #5dc247; color: white; border-radius: 4px; font-weight: bold;")
        self.search_btn.clicked.connect(self.on_search)
        self.search_btn.setEnabled(False)
        
        search_layout.addWidget(self.search_input)
        search_layout.addWidget(self.search_btn)
        
        layout.addWidget(search_group)
        
        # 3. 統計パネル
        stats_group = QGroupBox("ファイル統計")
        stats_layout = QVBoxLayout(stats_group)
        
        self.stat_files = QLabel("ファイル数: 0")
        self.stat_size = QLabel("総サイズ：0 B")
        self.stat_ext_count = QLabel("拡張子種類：0")
        self.stat_files.setStyleSheet("font-size: 14px; color: #333333; font-weight: bold;")
        self.stat_size.setStyleSheet("font-size: 14px; color: #333333;")
        self.stat_ext_count.setStyleSheet("font-size: 14px; color: #333333;")
        
        stats_layout.addWidget(self.stat_files)
        stats_layout.addWidget(self.stat_size)
        stats_layout.addWidget(self.stat_ext_count)
        
        layout.addWidget(stats_group)
        
        # 4. 表示設定
        display_group = QGroupBox("表示設定")
        display_layout = QVBoxLayout(display_group)
        
        self.display_mode_combo = QComboBox()
        self.display_mode_combo.addItems(["1 行で表示", "ファイル名で区切って表示"])
        self.display_mode_combo.setFixedWidth(150)
        self.display_mode_combo.setStyleSheet("border: 1px solid #cccccc; border-radius: 4px; padding: 6px;")
        self.display_mode_combo.currentTextChanged.connect(self.on_display_mode_changed)
        
        self.color_mode_combo = QComboBox()
        self.color_mode_combo.addItems(["無効", "拡張子色", "色パターン"])
        self.color_mode_combo.setFixedWidth(150)
        self.color_mode_combo.setStyleSheet("border: 1px solid #cccccc; border-radius: 4px; padding: 6px;")
        self.color_mode_combo.currentTextChanged.connect(self.on_color_mode_changed)
        
        # CSV エクスポート
        self.csv_btn = QPushButton("📄 CSV エクスポート")
        self.csv_btn.setFixedSize(150, 30)
        self.csv_btn.setStyleSheet("background-color: #5a9eff; color: white; border-radius: 4px;")
        self.csv_btn.clicked.connect(self.export_csv)
        self.csv_btn.setEnabled(False)
        
        # 一覧表示ボタン
        self.display_btn = QPushButton("📋 一覧を表示")
        self.display_btn.setFixedSize(150, 30)
        self.display_btn.setStyleSheet("background-color: #8a452a; color: white; border-radius: 4px;")
        self.display_btn.clicked.connect(self.show_file_list)
        self.display_btn.setEnabled(False)
        
        # クリアボタン
        self.clear_btn = QPushButton("🗑 一覧をクリア")
        self.clear_btn.setFixedSize(150, 30)
        self.clear_btn.setStyleSheet("background-color: #cc5555; color: white; border-radius: 4px;")
        self.clear_btn.clicked.connect(self.clear_list)
        
        display_layout.addWidget(self.display_mode_combo)
        display_layout.addWidget(self.color_mode_combo)
        display_layout.addWidget(self.display_btn)
        display_layout.addWidget(self.csv_btn)
        display_layout.addWidget(self.clear_btn)
        
        display_group.setLayout(display_layout)
        layout.addWidget(display_group)
        
        # 5. 表示エリア
        self.display_layout = QVBoxLayout()
        self.display_area = QWidget()
        self.display_layout.addWidget(self.display_area)
        layout.addLayout(self.display_layout)
        
        # 初期設定
        self.display_mode_combo.currentTextChanged.connect(self.on_display_mode_changed)
        self.color_mode_combo.currentTextChanged.connect(self.on_color_mode_changed)
    
    def on_folder_selected(self):
        """フォルダ選択がされたら実行"""
        folder_path = self.file_analyzer.select_folder()
        if folder_path:
            self.folder_label.setText(f"選択フォルダ:\n{folder_path}")
            self.folder_label.setStyleSheet("color: #339966; font-weight: bold; padding: 10px;")
            self.folder_btn.setEnabled(False)
            self.update_stats()
            self.csv_btn.setEnabled(True)
            self.display_btn.setEnabled(True)
    
    def update_stats(self):
        """統計情報の更新"""
        self.stat_files.setText(f"ファイル数：{self.file_analyzer.total_files}")
        self.stat_size.setText(f"総サイズ：{self.file_analyzer.total_size} B")
        self.stat_ext_count.setText(f"拡張子種類：{len(self.file_analyzer.file_extensions)}")
    
    def on_search(self):
        """検索ボタンが押されたら実行"""
        keyword = self.search_input.text().strip()
        if not keyword:
            self.search_input.clear()
            return
        
        results = self.file_analyzer.filter_by_keyword(keyword)
        self.filter_results(results)
    
    def filter_results(self, results):
        """検索結果を表示"""
        self.file_analyzer.filtered_files = results
        self.file_analyzer.filtered_extensions = set()
        for f in results:
            if f["extension"] not in self.file_analyzer.filtered_extensions:
                self.file_analyzer.filtered_extensions.add(f["extension"])
        
        self.update_stats_filtered()
    
    def update_stats_filtered(self):
        """フィルタ後の統計更新"""
        total_files = sum(len(files) for ext, files in self.file_analyzer.filtered_groups.items())
        total_size = sum(
            sum(f["size"] for f in files)
            for ext, files in self.file_analyzer.filtered_groups.items()
        )
        self.stat_files.setText(f"ファイル数：{total_files}")
        self.stat_size.setText(f"総サイズ：{total_size} B")
    
    def on_display_mode_changed(self):
        """表示モードの切り替え"""
        mode = self.display_mode_combo.currentText()
        if mode == "1 行で表示":
            self.display_layout.setSpacing(8)
        else:
            self.display_layout.setSpacing(12)
    
    def on_color_mode_changed(self):
        """色分けモードの切り替え"""
        mode = self.color_mode_combo.currentText()
        if mode == "無効":
            self.color_mode = "none"
        elif mode == "拡張子色":
            self.color_mode = "extension"
        else:
            self.color_mode = "pattern"
    
    def show_file_list(self):
        """ファイル一覧を表示"""
        self.display_layout.removeWidget(self.display_area)
        self.display_area = QWidget()
        self.display_layout.addWidget(self.display_area)
        self.display_layout.setContentsMargins(0, 0, 0, 0)
        
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["拡張子", "ファイル数", "ファイルサイズ合計", "ファイル名一覧", ""])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setRowCount(0)
        
        palette = COLOR_PALETTE
        row = 0
        group_data = self.file_analyzer.get_extension_groups()
        
        # 拡張子をソート（. の付いた形）
        sorted_groups = {f".{k}": v for k, v in group_data.items()}
        sorted_groups = dict(sorted(sorted_groups.items()))
        
        for ext, data in sorted_groups.items():
            row += 1
            self.table.insertRow(row - 1)
            
            # 拡張子
            item_ext = QTableWidgetItem(ext.lstrip("."))
            item_ext.setTextAlignment(Qt.AlignCenter)
            
            # ファイル数
            item_count = QTableWidgetItem(str(data["count"]))
            item_count.setTextAlignment(Qt.AlignCenter)
            
            # サイズ
            item_size = QTableWidgetItem(f"{data['total_size_display']}")
            item_size.setTextAlignment(Qt.AlignRight)
            
            # ファイル名一覧
            names_str = "\\n".join(f"{f['name']} ({f['size_display']})" for f in data["files"])
            item_names = QTableWidgetItem(names_str)
            item_names.setTextAlignment(Qt.AlignLeft)
            
            # 背景色とフォント色を設定
            self.set_row_color(item_ext.row(), ext, palette)
            
            # 表示
            self.table.setItem(0, item_ext)
            self.table.setItem(1, item_count)
            self.table.setItem(2, item_size)
            self.table.setItem(3, item_names)
    
    def set_row_color(self, row_index, ext, palette):
        """行の色設定"""
        if self.color_mode == "none":
            return
        
        if self.color_mode == "extension":
            if ext.startswith(".py"):
                bg = QColor("#e8f5e9", 0.95)
                fg = QColor("#2e7d32")
            elif ext.startswith(".txt"):
                bg = QColor("#f5f5f5", 0.95)
                fg = QColor("#333333")
            elif ext.startswith(".md"):
                bg = QColor("#e3f2fd", 0.95)
                fg = QColor("#1565c0")
            elif ext.startswith(".json"):
                bg = QColor("#fffde7", 0.95)
                fg = QColor("#fbc02d")
            elif ext.startswith(".html"):
                bg = QColor("#fce4ec", 0.95)
                fg = QColor("#c2185b")
            elif ext.startswith(".js"):
                bg = QColor("#f3e5f5", 0.95)
                fg = QColor("#7b1fa2")
            else:
                bg = QColor("#f5f5f5", 0.98)
                fg = QColor("#333333")
        else:
            # 色パターン：白黒交互
            bg = QColor("#ffffff" if row_index % 2 == 0 else "#f0f0f0")
            fg = QColor("#333333")
        
        bg = bg.toString()
        fg = fg.toString()
        
        row_style = f"background-color: {bg}; color: {fg}; font-weight: bold; border-bottom: 1px solid #eee;"
        self.table.item(0, item_ext.row()).setForeground(fg)
        self.table.item(1, item_ext.row()).setForeground(fg)
        self.table.item(2, item_ext.row()).setForeground(fg)
        self.table.item(3, item_ext.row()).setForeground(fg)
    
    def export_csv(self):
        """CSV エクスポート"""
        if not self.file_analyzer.file_extensions:
            return
        
        folder_path = self.file_analyzer.folder_path
        filename, _ = QFileDialog.getSaveFileName(None, "CSV を保存", folder_path, "CSV ファイル (*.csv);;Text Files (*.txt)")
        
        if not filename:
            return
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write("拡張子，ファイル数，サイズ合計\\n")
            for ext, data in self.file_analyzer.get_extension_groups().items():
                f.write(f"{ext},{data['count']},{data['total_size']}\\n")
        
        QMessageBox.information(self, "完了", f"{filename} にエクスポートしました")
    
    def clear_list(self):
        """一覧をクリア"""
        self.display_layout.removeWidget(self.display_area)
        self.display_area = QWidget()
        self.display_layout.addWidget(self.display_area)
        
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["拡張子", "ファイル数", "ファイルサイズ合計", "ファイル名一覧", ""])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setRowCount(0)
        self.table.horizontalHeader().setStyleSheet("background-color: #f8f8f8; color: #333333; font-weight: bold; padding: 8px;")
        
        self.display_layout.addWidget(self.table)
        self.csv_btn.setEnabled(False)
    
    def filter_results(self, results):
        """検索結果を表示"""
        self.file_analyzer.filtered_files = results
        self.file_analyzer.filtered_extensions = set()
        for f in results:
            if f["extension"] not in self.file_analyzer.filtered_extensions:
                self.file_analyzer.filtered_extensions.add(f["extension"])
        
        self.update_stats_filtered()
    
    def update_stats_filtered(self):
        """フィルタ後の統計更新"""
        total_files = sum(len(files) for ext, files in self.file_analyzer.filtered_groups.items())
        total_size = sum(
            sum(f["size"] for f in files)
            for ext, files in self.file_analyzer.filtered_groups.items()
        )
        self.stat_files.setText(f"ファイル数：{total_files}")
        self.stat_size.setText(f"総サイズ：{total_size} B")


def main():
    app = QApplication(sys.argv)
    window = FileExplorerUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()