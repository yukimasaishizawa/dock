using System.Runtime.InteropServices;
using System.Windows;
using System.Windows.Threading;

namespace ShopActivityPopup;

public partial class MainWindow : Window
{
    private readonly AppConfig _config;
    private readonly DispatcherTimer _idleTimer;
    private bool _manualClose;

    public MainWindow(AppConfig config)
    {
        InitializeComponent();
        _config = config;
        var workArea = SystemParameters.WorkArea;
        Width = config.PopupWidth > 0 ? config.PopupWidth : Math.Max(320, (int)(workArea.Width * .8));
        Height = config.PopupHeight > 0 ? config.PopupHeight : Math.Max(240, (int)(workArea.Height * .8));
        Left = workArea.Left + (workArea.Width - Width) / 2;
        Top = workArea.Top + (workArea.Height - Height) / 2;

        _idleTimer = new DispatcherTimer { Interval = TimeSpan.FromSeconds(1) };
        _idleTimer.Tick += (_, _) =>
        {
            if (GetIdleMilliseconds() >= _config.CloseAfterIdleSeconds * 1000L)
                Close();
        };
        Closed += (_, _) => _idleTimer.Stop();
    }

    public async Task OpenUrlAsync()
    {
        try
        {
            await Browser.EnsureCoreWebView2Async();
            Browser.CoreWebView2.Navigate(_config.TargetUrl);
            Show();
            Activate();
            _idleTimer.Start();
        }
        catch (Exception ex)
        {
            MessageBox.Show($"WebView2を初期化できませんでした。\n{ex.Message}", "エラー", MessageBoxButton.OK, MessageBoxImage.Error);
            Close();
        }
    }

    private void BackButton_Click(object sender, RoutedEventArgs e)
    {
        if (Browser.CanGoBack) Browser.GoBack();
    }

    private void ForwardButton_Click(object sender, RoutedEventArgs e)
    {
        if (Browser.CanGoForward) Browser.GoForward();
    }

    private void CloseButton_Click(object sender, RoutedEventArgs e)
    {
        _manualClose = true;
        Close();
    }

    public bool WasManuallyClosed => _manualClose;

    private static long GetIdleMilliseconds()
    {
        var info = new LASTINPUTINFO { cbSize = (uint)Marshal.SizeOf<LASTINPUTINFO>() };
        if (!GetLastInputInfo(ref info)) return 0;
        uint now = unchecked((uint)Environment.TickCount64);
        return unchecked(now - info.dwTime);
    }

    [StructLayout(LayoutKind.Sequential)] private struct LASTINPUTINFO { public uint cbSize; public uint dwTime; }
    [DllImport("user32.dll")] [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool GetLastInputInfo(ref LASTINPUTINFO plii);
}
