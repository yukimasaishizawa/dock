using System.ComponentModel;
using System.Runtime.InteropServices;

namespace ShopActivityPopup;

/// <summary>
/// キーの内容やマウス座標は読み取らず、入力イベントの発生だけを通知するフック。
/// </summary>
public sealed class GlobalInputHook : IDisposable
{
    private const int WH_KEYBOARD_LL = 13;
    private const int WH_MOUSE_LL = 14;
    private const int WM_MOUSEMOVE = 0x0200;
    private const int WM_LBUTTONDOWN = 0x0201;
    private const int WM_RBUTTONDOWN = 0x0204;
    private const int WM_MBUTTONDOWN = 0x0207;

    private readonly LowLevelProc _keyboardProc;
    private readonly LowLevelProc _mouseProc;
    private nint _keyboardHook;
    private nint _mouseHook;
    private bool _disposed;

    public event EventHandler? InputDetected;

    public GlobalInputHook()
    {
        // デリゲートをフィールドに保持し、GCによるコールバック消失を防ぐ。
        _keyboardProc = KeyboardCallback;
        _mouseProc = MouseCallback;
    }

    public void Start()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_keyboardHook != 0 || _mouseHook != 0) return;

        using var process = System.Diagnostics.Process.GetCurrentProcess();
        using var module = process.MainModule ?? throw new InvalidOperationException("プロセスモジュールを取得できません。");
        var moduleHandle = GetModuleHandle(module.ModuleName);

        _keyboardHook = SetWindowsHookEx(WH_KEYBOARD_LL, _keyboardProc, moduleHandle, 0);
        _mouseHook = SetWindowsHookEx(WH_MOUSE_LL, _mouseProc, moduleHandle, 0);
        if (_keyboardHook == 0 || _mouseHook == 0)
        {
            Stop();
            throw new Win32Exception(Marshal.GetLastWin32Error(), "グローバル入力フックを登録できません。");
        }
    }

    public void Stop()
    {
        if (_keyboardHook != 0) { UnhookWindowsHookEx(_keyboardHook); _keyboardHook = 0; }
        if (_mouseHook != 0) { UnhookWindowsHookEx(_mouseHook); _mouseHook = 0; }
    }

    private nint KeyboardCallback(int nCode, nint wParam, nint lParam)
    {
        if (nCode >= 0) InputDetected?.Invoke(this, EventArgs.Empty);
        return CallNextHookEx(0, nCode, wParam, lParam);
    }

    private nint MouseCallback(int nCode, nint wParam, nint lParam)
    {
        if (nCode >= 0 && (wParam == WM_MOUSEMOVE || wParam == WM_LBUTTONDOWN ||
                           wParam == WM_RBUTTONDOWN || wParam == WM_MBUTTONDOWN))
        {
            InputDetected?.Invoke(this, EventArgs.Empty);
        }
        return CallNextHookEx(0, nCode, wParam, lParam);
    }

    public void Dispose()
    {
        if (_disposed) return;
        Stop();
        _disposed = true;
        GC.SuppressFinalize(this);
    }

    private delegate nint LowLevelProc(int nCode, nint wParam, nint lParam);

    [DllImport("user32.dll", SetLastError = true)]
    private static extern nint SetWindowsHookEx(int idHook, LowLevelProc lpfn, nint hMod, uint dwThreadId);

    [DllImport("user32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool UnhookWindowsHookEx(nint hhk);

    [DllImport("user32.dll")]
    private static extern nint CallNextHookEx(nint hhk, int nCode, nint wParam, nint lParam);

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern nint GetModuleHandle(string? lpModuleName);
}
