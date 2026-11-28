using System.Diagnostics;
using System.ComponentModel;
using System.Drawing;
using System.Security.Principal;
using System.Windows;
using System.Windows.Threading;
using Forms = System.Windows.Forms;

namespace ShopActivityPopup;

public partial class App : Application
{
    private AppConfig _config = null!;
    private GlobalInputHook _inputHook = null!;
    private Forms.NotifyIcon _trayIcon = null!;
    private HotkeyWindow? _hotkeyWindow;
    private MainWindow? _popup;
    private bool _showing;
    private bool _suppressPopupRestart;
    private DispatcherTimer? _cooldownTimer;

    protected override void OnStartup(StartupEventArgs e)
    {
        base.OnStartup(e);
        _config = AppConfig.Load();

        // Ctrl+Sから起動された設定専用プロセス。通常の常駐処理は開始しない。
        if (e.Args.Contains("--settings", StringComparer.OrdinalIgnoreCase))
        {
            var settings = new SettingsWindow(_config);
            settings.ShowDialog();
            Shutdown(0);
            return;
        }

        ConfigureTrayIcon();
        _inputHook = new GlobalInputHook();
        _inputHook.InputDetected += (_, _) => Dispatcher.BeginInvoke(ShowPopup);
        _hotkeyWindow = new HotkeyWindow();
        _hotkeyWindow.CtrlSPressed += (_, _) => Dispatcher.BeginInvoke(ShowSettingsWithUac);
        _hotkeyWindow.Show();
        StartMonitoring();
    }

    private void ConfigureTrayIcon()
    {
        var menu = new Forms.ContextMenuStrip();
        menu.Items.Add("今すぐ表示", null, (_, _) => Dispatcher.InvokeAsync(ShowPopup));
        menu.Items.Add("再監視", null, (_, _) => StartMonitoring());
        menu.Items.Add(new Forms.ToolStripSeparator());
        menu.Items.Add("終了", null, (_, _) => ShutdownApp());
        _trayIcon = new Forms.NotifyIcon { Icon = SystemIcons.Application, Text = "Web Popup 常駐中", ContextMenuStrip = menu, Visible = true };
        _trayIcon.DoubleClick += (_, _) => Dispatcher.InvokeAsync(ShowPopup);
    }

    private void StartMonitoring()
    {
        _cooldownTimer?.Stop();
        if (!_showing) _inputHook.Start();
    }

    private async void ShowPopup()
    {
        if (_showing) return;
        _cooldownTimer?.Stop();
        _showing = true;
        _inputHook.Stop();
        _popup = new MainWindow(_config);
        _popup.Closed += (_, _) =>
        {
            var manual = _popup?.WasManuallyClosed == true;
            _popup = null;
            _showing = false;
            if (_suppressPopupRestart) return;
            if (manual) StartCooldown(); else StartMonitoring();
        };
        await _popup.OpenUrlAsync();
    }

    private void ShowSettingsWithUac()
    {
        _inputHook.Stop();
        _suppressPopupRestart = true;
        _popup?.Close();
        _popup = null;
        _showing = false;

        if (IsAdministrator())
        {
            ShowSettingsDialog();
        }
        else
        {
            try
            {
                // Windows標準のUAC画面を表示する。アプリ側で管理者パスワードを取得・保存しない。
                var startInfo = new ProcessStartInfo(Environment.ProcessPath!)
                {
                    Arguments = "--settings",
                    UseShellExecute = true,
                    Verb = "runas"
                };
                var settingsProcess = Process.Start(startInfo);
                if (settingsProcess is not null)
                {
                    settingsProcess.EnableRaisingEvents = true;
                    settingsProcess.Exited += (_, _) => Dispatcher.BeginInvoke(StartMonitoring);
                }
                else
                {
                    StartMonitoring();
                }
            }
            catch (Win32Exception)
            {
                // UACでキャンセルされた場合は設定を変更せず、通常監視へ戻る。
                StartMonitoring();
            }
        }

        _suppressPopupRestart = false;
        if (IsAdministrator()) StartMonitoring();
    }

    private void ShowSettingsDialog()
    {
        var settings = new SettingsWindow(_config);
        settings.ShowDialog();
        _config = AppConfig.Load();
    }

    private static bool IsAdministrator()
    {
        using var identity = WindowsIdentity.GetCurrent();
        return new WindowsPrincipal(identity).IsInRole(WindowsBuiltInRole.Administrator);
    }

    private void StartCooldown()
    {
        _inputHook.Stop();
        _cooldownTimer = new DispatcherTimer { Interval = TimeSpan.FromSeconds(1) };
        _cooldownTimer.Tick += (_, _) => { if (GetIdleMilliseconds() >= 30_000) StartMonitoring(); };
        _cooldownTimer.Start();
    }

    private static long GetIdleMilliseconds()
    {
        var info = new LASTINPUTINFO { cbSize = (uint)System.Runtime.InteropServices.Marshal.SizeOf<LASTINPUTINFO>() };
        if (!GetLastInputInfo(ref info)) return 0;
        uint now = unchecked((uint)Environment.TickCount64);
        return unchecked(now - info.dwTime);
    }

    private void ShutdownApp()
    {
        _inputHook?.Dispose();
        _hotkeyWindow?.Close();
        if (_trayIcon is not null) { _trayIcon.Visible = false; _trayIcon.Dispose(); }
        _popup?.Close();
        Shutdown(0);
    }

    protected override void OnExit(ExitEventArgs e)
    {
        _inputHook?.Dispose();
        _hotkeyWindow?.Close();
        if (_trayIcon is not null) { _trayIcon.Visible = false; _trayIcon.Dispose(); }
        base.OnExit(e);
    }

    private struct LASTINPUTINFO { public uint cbSize; public uint dwTime; }
    [System.Runtime.InteropServices.DllImport("user32.dll")]
    [return: System.Runtime.InteropServices.MarshalAs(System.Runtime.InteropServices.UnmanagedType.Bool)]
    private static extern bool GetLastInputInfo(ref LASTINPUTINFO plii);
}
