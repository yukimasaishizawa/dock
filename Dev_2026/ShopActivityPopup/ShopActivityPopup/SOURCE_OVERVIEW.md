# ソース構成

- `App.xaml.cs`: タスクトレイ常駐、入力監視、Ctrl+S、UACを使った設定専用プロセスの起動、30秒クールダウンを管理します。
- `GlobalInputHook.cs`: Win32の低レベルキーボード／マウスフックです。入力内容ではなく、イベント発生だけを通知します。
- `HotkeyWindow.cs`: `RegisterHotKey`でCtrl+Sだけを受け取る不可視ウィンドウです。
- `MainWindow.xaml` / `MainWindow.xaml.cs`: 丸角の枠、WebView2、サイト上に浮かぶ戻る・進む・閉じるの3ボタン、中央配置、3分アイドル終了を実装します。
- `SettingsWindow.xaml` / `SettingsWindow.xaml.cs`: URL、幅、高さ、無操作秒数を編集し、保存時に入力値を検証します。
- `AppConfig.cs`: `appsettings.json`の読み書きとURLのhttp/https検証を行います。
- `appsettings.json`: 初期URL、表示サイズ、アイドル秒数を保持します。

Ctrl+Sの設定プロセスは`--settings`引数で起動され、通常の常駐フックやタスクトレイを開始しません。管理者資格情報の取得はWindowsのUACに任せ、アプリ側では扱いません。
