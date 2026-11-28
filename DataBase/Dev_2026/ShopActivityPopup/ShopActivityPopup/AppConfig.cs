using System.Text.Json;

namespace ShopActivityPopup;

public sealed class AppConfig
{
    public string TargetUrl { get; set; } = "https://example.com";
    // 0は「画面サイズの80%」を意味する。
    public int PopupWidth { get; set; }
    public int PopupHeight { get; set; }
    public int CloseAfterIdleSeconds { get; set; } = 180;

    public static AppConfig Load()
    {
        var path = Path.Combine(AppContext.BaseDirectory, "appsettings.json");
        if (!File.Exists(path)) return new AppConfig();
        try
        {
            var config = JsonSerializer.Deserialize<AppConfig>(File.ReadAllText(path)) ?? new AppConfig();
            if (!Uri.TryCreate(config.TargetUrl, UriKind.Absolute, out var uri) ||
                (uri.Scheme != Uri.UriSchemeHttp && uri.Scheme != Uri.UriSchemeHttps))
                config.TargetUrl = "https://example.com";
            config.CloseAfterIdleSeconds = Math.Max(1, config.CloseAfterIdleSeconds);
            config.PopupWidth = Math.Max(0, config.PopupWidth);
            config.PopupHeight = Math.Max(0, config.PopupHeight);
            return config;
        }
        catch { return new AppConfig(); }
    }

    public void Save()
    {
        var path = Path.Combine(AppContext.BaseDirectory, "appsettings.json");
        var options = new JsonSerializerOptions { WriteIndented = true };
        File.WriteAllText(path, JsonSerializer.Serialize(this, options));
    }
}
