using System.Runtime.InteropServices;
using System.Windows;
using System.Windows.Interop;

namespace ShopActivityPopup;

public sealed class HotkeyWindow : Window
{
    private const int HOTKEY_ID = 0x5201;
    private const int WM_HOTKEY = 0x0312;
    private const uint MOD_CONTROL = 0x0002;
    private const uint VK_S = 0x53;
    private HwndSource? _source;
    public event EventHandler? CtrlSPressed;

    public HotkeyWindow()
    {
        Width = 0; Height = 0; ShowInTaskbar = false; WindowStyle = WindowStyle.None;
        Opacity = 0; ShowActivated = false;
        SourceInitialized += (_, _) =>
        {
            _source = (HwndSource)PresentationSource.FromVisual(this)!;
            _source.AddHook(WndProc);
            RegisterHotKey(_source.Handle, HOTKEY_ID, MOD_CONTROL, VK_S);
        };
        Closed += (_, _) =>
        {
            if (_source is not null) UnregisterHotKey(_source.Handle, HOTKEY_ID);
        };
    }

    private nint WndProc(nint hwnd, int msg, nint wParam, nint lParam, ref bool handled)
    {
        if (msg == WM_HOTKEY && wParam.ToInt32() == HOTKEY_ID)
        {
            CtrlSPressed?.Invoke(this, EventArgs.Empty);
            handled = true;
        }
        return 0;
    }

    [DllImport("user32.dll", SetLastError = true)] private static extern bool RegisterHotKey(nint hWnd, int id, uint fsModifiers, uint vk);
    [DllImport("user32.dll", SetLastError = true)] private static extern bool UnregisterHotKey(nint hWnd, int id);
}
