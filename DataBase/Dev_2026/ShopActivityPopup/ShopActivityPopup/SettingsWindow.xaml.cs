using System.Windows;

namespace ShopActivityPopup;

public partial class SettingsWindow : Window
{
    private readonly AppConfig _config;
    public bool Saved { get; private set; }

    public SettingsWindow(AppConfig config)
    {
        InitializeComponent();
        _config = config;
        var workArea = SystemParameters.WorkArea;
        UrlBox.Text = config.TargetUrl;
        WidthBox.Text = (config.PopupWidth > 0 ? config.PopupWidth : (int)(workArea.Width * .8)).ToString();
        HeightBox.Text = (config.PopupHeight > 0 ? config.PopupHeight : (int)(workArea.Height * .8)).ToString();
        IdleBox.Text = config.CloseAfterIdleSeconds.ToString();
    }

    private void Save_Click(object sender, RoutedEventArgs e)
    {
        if (!Uri.TryCreate(UrlBox.Text.Trim(), UriKind.Absolute, out var uri) ||
            (uri.Scheme != Uri.UriSchemeHttp && uri.Scheme != Uri.UriSchemeHttps) ||
            !int.TryParse(WidthBox.Text, out var width) || width < 320 ||
            !int.TryParse(HeightBox.Text, out var height) || height < 240 ||
            !int.TryParse(IdleBox.Text, out var idle) || idle < 1)
        {
            MessageBox.Show("URLはhttp/https、幅は320以上、高さは240以上、秒数は1以上で入力してください。", "入力エラー", MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }
        _config.TargetUrl = uri.ToString();
        _config.PopupWidth = width;
        _config.PopupHeight = height;
        _config.CloseAfterIdleSeconds = idle;
        try { _config.Save(); }
        catch (Exception ex)
        {
            MessageBox.Show($"設定を保存できませんでした。\n{ex.Message}", "保存エラー", MessageBoxButton.OK, MessageBoxImage.Error);
            return;
        }
        Saved = true;
        Close();
    }

    private void Cancel_Click(object sender, RoutedEventArgs e) => Close();
}
