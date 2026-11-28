#Requires AutoHotkey v2.0
#SingleInstance Off
Persistent

; ==============================================================================
; ShopActivityPopup - AutoHotkey v2 実装版
; 仕様書準拠: 常駐監視 / 島型ボタン / 30秒クールダウン / Ctrl+S UAC設定分離
; ==============================================================================

; グローバル設定・状態変数（初期URL組み込み済み）
global ConfigFile := A_ScriptDir "\appsettings.json"
global EdgeProfileDir := A_Temp "\ShopActivityPopup_EdgeProfile"
global Config := {
    TargetUrl: "https://nojimapc2026.github.io/Optimal_PC_proposer_customer/#",
    PopupWidth: 0,
    PopupHeight: 0,
    CloseAfterIdleSeconds: 180
}

; プロセスモード判定（/settings 引数がある場合は設定専用プロセスとして起動）
global IsSettingsProcess := false
for arg in A_Args {
    if (arg = "/settings") {
        IsSettingsProcess := true
        break
    }
}

; ==============================================================================
; エントリーポイント分岐（14.4 セキュリティ仕様: 常駐プロセスと設定プロセスの分離）
; ==============================================================================
if (IsSettingsProcess) {
    ; 設定専用プロセス: トレイアイコン非表示・入力フックなし
    A_IconHidden := true
    LoadConfig()
    ShowSettingsWindow()
    return
}

; --- 以下、通常常駐プロセス ---
; 多重起動防止（設定プロセス以外の常駐プロセスが既に存在する場合は終了）
EnforceSingleResidentInstance()

; 低レベルキーボード・マウスフックの有効化（入力内容の記録は行わない）
InstallKeybdHook()
InstallMouseHook()

; タスクトレイ設定（仕様書14.2: トレイメニューから設定画面は開けない仕様）
A_IconTip := "ShopActivityPopup - 待機中"
A_TrayMenu.Delete()
A_TrayMenu.Add("終了", (*) => ExitApp())

; 初期設定ファイルの読み込み（なければ初期値で作成）
if !FileExist(ConfigFile) {
    SaveConfig(Config)
} else {
    LoadConfig()
}
global LastConfigModTime := FileGetTime(ConfigFile, "M")

; 状態管理定数
global STATE_MONITORING := 1  ; 入力監視・待機中
global STATE_SHOWING    := 2  ; サイト表示中
global STATE_COOLDOWN   := 3  ; 手動閉じる後のクールダウン中（30秒無操作待ち）

global CurrentState := STATE_MONITORING
global LastIdleCheck := A_TimeIdlePhysical
global BrowserHwnd := 0
global LeftIslandGui := ""
global RightIslandGui := ""

; グローバルホットキー: Ctrl+S のみで設定画面を起動（UAC昇格）
^s::OpenSettingsWithUAC()

; メイン監視ループ（150ms間隔）
SetTimer(MainLoop, 150)
return


; ==============================================================================
; メイン監視・状態遷移ループ
; ==============================================================================
MainLoop() {
    global CurrentState, LastIdleCheck, BrowserHwnd, Config, ConfigFile, LastConfigModTime

    ; 1. 設定ファイルの更新検知（設定プロセスでの保存後に待機状態へ戻す）
    if FileExist(ConfigFile) {
        currentModTime := FileGetTime(ConfigFile, "M")
        if (currentModTime != LastConfigModTime) {
            LastConfigModTime := currentModTime
            LoadConfig()
            ClosePopup()
            CurrentState := STATE_MONITORING
            LastIdleCheck := A_TimeIdlePhysical
            A_IconTip := "ShopActivityPopup - 待機中"
            return
        }
    }

    currentIdle := A_TimeIdlePhysical

    ; 2. 状態ごとの処理
    if (CurrentState = STATE_MONITORING) {
        ; 物理的なキー打鍵・マウス移動・クリックが発生すると A_TimeIdlePhysical がリセット（減少）する
        if (currentIdle < LastIdleCheck) {
            CurrentState := STATE_SHOWING
            A_IconTip := "ShopActivityPopup - 表示中"
            ShowPopup()
        }
        LastIdleCheck := currentIdle

    } else if (CurrentState = STATE_SHOWING) {
        ; ブラウザが外部操作（Alt+F4等）で閉じられた場合の検知
        if (!BrowserHwnd || !WinExist("ahk_id " BrowserHwnd)) {
            ClosePopup()
            EnterCooldown()
            return
        }

        ; 島型ボタンの位置・最前面同期
        SyncIslandsPosition()

        ; 指定秒数以上の無操作で自動終了
        if (currentIdle >= Config.CloseAfterIdleSeconds * 1000) {
            ClosePopup()
            CurrentState := STATE_MONITORING
            LastIdleCheck := A_TimeIdlePhysical
            A_IconTip := "ShopActivityPopup - 待機中"
        }

    } else if (CurrentState = STATE_COOLDOWN) {
        ; 仕様書14.1 / 15.3: 手動閉じる後、30秒連続アイドル後に入力監視へ復帰
        if (currentIdle >= 30000) {
            CurrentState := STATE_MONITORING
            LastIdleCheck := A_TimeIdlePhysical
            A_IconTip := "ShopActivityPopup - 待機中"
        }
    }
}


; ==============================================================================
; ポップアップ表示（Edge フレームレス化 + 丸角 + 島型ボタン）
; ==============================================================================
ShowPopup() {
    global Config, BrowserHwnd, EdgeProfileDir

    edgePath := FindEdgePath()
    if (edgePath = "") {
        MsgBox("Microsoft Edge が見つかりません。Web表示機能を初期化できません。", "エラー", "Iconx")
        CurrentState := STATE_MONITORING
        return
    }

    ; 既存のEdgeウィンドウ一覧を記録（新規ウィンドウ特定用）
    existingHwnds := Map()
    for hwnd in WinGetList("ahk_exe msedge.exe") {
        existingHwnds[hwnd] := true
    }

    ; 位置・サイズの計算（メインディスプレイの作業領域中央・初期値80%）
    MonitorGetWorkArea(, &waLeft, &waTop, &waRight, &waBottom)
    waWidth := waRight - waLeft
    waHeight := waBottom - waTop

    popupW := (Config.PopupWidth > 0) ? Config.PopupWidth : Round(waWidth * 0.8)
    popupH := (Config.PopupHeight > 0) ? Config.PopupHeight : Round(waHeight * 0.8)
    popupX := waLeft + Round((waWidth - popupW) / 2)
    popupY := waTop + Round((waHeight - popupH) / 2)

    ; 専用プロファイル・アプリモードでEdgeを起動
    cmd := Format('"{1}" --app="{2}" --user-data-dir="{3}" --no-first-run --no-default-browser-check --disable-features=Translate --window-position={4},{5} --window-size={6},{7}',
        edgePath, Config.TargetUrl, EdgeProfileDir, popupX, popupY, popupW, popupH)
    Run(cmd)

    ; 新しいEdgeウィンドウが出現するまで待機（最大5秒）
    targetHwnd := 0
    startTick := A_TickCount
    while (A_TickCount - startTick < 5000) {
        for hwnd in WinGetList("ahk_exe msedge.exe") {
            if (!existingHwnds.Has(hwnd) && WinGetTitle("ahk_id " hwnd) != "") {
                targetHwnd := hwnd
                break 2
            }
        }
        Sleep(50)
    }

    if (!targetHwnd) {
        CurrentState := STATE_MONITORING
        return
    }

    BrowserHwnd := targetHwnd

    ; タイトルバーとサイズ変更枠を除去（WS_CAPTION=0xC00000, WS_THICKFRAME=0x40000）
    try {
        WinSetStyle("-0xC40000", "ahk_id " BrowserHwnd)
        WinMove(popupX, popupY, popupW, popupH, "ahk_id " BrowserHwnd)
        ApplyRoundCorners(BrowserHwnd, popupW, popupH, 16)
        WinActivate("ahk_id " BrowserHwnd)
    }

    ; 上部左右の島型ボタンを生成
    CreateIslandButtons(popupX, popupY, popupW)
}


; ==============================================================================
; 島型ボタン（左上: 戻る・進む / 右上: 閉じる）の生成と制御
; ==============================================================================
CreateIslandButtons(popupX, popupY, popupW) {
    global LeftIslandGui, RightIslandGui, BrowserHwnd

    DestroyIslands()

    margin := 14

    ; --- 左上: 戻る・進む 島型ボタン ---
    LeftIslandGui := Gui("-Caption +ToolWindow +AlwaysOnTop")
    LeftIslandGui.BackColor := "F4F5F7"
    LeftIslandGui.SetFont("s10 Bold", "Yu Gothic UI")
    LeftIslandGui.MarginX := 6
    LeftIslandGui.MarginY := 5

    btnBack := LeftIslandGui.Add("Button", "w68 h28", "← 戻る")
    btnFwd  := LeftIslandGui.Add("Button", "x+6 w68 h28", "進む →")

    btnBack.OnEvent("Click", (*) => NavigateBrowser("Back"))
    btnFwd.OnEvent("Click", (*) => NavigateBrowser("Forward"))

    LeftIslandGui.Show(Format("x{1} y{2} NoActivate", popupX + margin, popupY + margin))
    LeftIslandGui.GetPos(, , &lw, &lh)
    ApplyRoundCorners(LeftIslandGui.Hwnd, lw, lh, 12)
    SetWindowOwner(LeftIslandGui.Hwnd, BrowserHwnd)

    ; --- 右上: 閉じる 島型ボタン ---
    RightIslandGui := Gui("-Caption +ToolWindow +AlwaysOnTop")
    RightIslandGui.BackColor := "F4F5F7"
    RightIslandGui.SetFont("s11 Bold", "Yu Gothic UI")
    RightIslandGui.MarginX := 6
    RightIslandGui.MarginY := 5

    btnClose := RightIslandGui.Add("Button", "w36 h28", "✕")
    btnClose.OnEvent("Click", (*) => ManualClosePopup())

    RightIslandGui.Show(Format("x{1} y{2} NoActivate", popupX + popupW - 48 - margin, popupY + margin))
    RightIslandGui.GetPos(, , &rw, &rh)
    ApplyRoundCorners(RightIslandGui.Hwnd, rw, rh, 12)
    SetWindowOwner(RightIslandGui.Hwnd, BrowserHwnd)
}

SyncIslandsPosition() {
    global BrowserHwnd, LeftIslandGui, RightIslandGui
    if (!BrowserHwnd || !WinExist("ahk_id " BrowserHwnd))
        return

    try {
        WinGetPos(&bx, &by, &bw, &bh, "ahk_id " BrowserHwnd)
        margin := 14
        if IsObject(LeftIslandGui) {
            LeftIslandGui.Move(bx + margin, by + margin)
        }
        if IsObject(RightIslandGui) {
            RightIslandGui.GetPos(, , &rw)
            RightIslandGui.Move(bx + bw - rw - margin, by + margin)
        }
    }
}

NavigateBrowser(direction) {
    global BrowserHwnd
    if (!BrowserHwnd || !WinExist("ahk_id " BrowserHwnd))
        return
    WinActivate("ahk_id " BrowserHwnd)
    Sleep(30)
    if (direction = "Back") {
        Send("!{Left}")
    } else if (direction = "Forward") {
        Send("!{Right}")
    }
}

ManualClosePopup() {
    ClosePopup()
    EnterCooldown()
}

EnterCooldown() {
    global CurrentState
    CurrentState := STATE_COOLDOWN
    A_IconTip := "ShopActivityPopup - クールダウン中(30秒)"
}

ClosePopup() {
    global BrowserHwnd, EdgeProfileDir
    DestroyIslands()

    if (BrowserHwnd && WinExist("ahk_id " BrowserHwnd)) {
        try WinClose("ahk_id " BrowserHwnd)
    }
    BrowserHwnd := 0

    ; 専用プロファイルで起動している残存Edgeプロセスを確実に終了
    for proc in ComObjGet("winmgmts:").ExecQuery("SELECT ProcessId, CommandLine FROM Win32_Process WHERE Name = 'msedge.exe'") {
        if InStr(proc.CommandLine, "ShopActivityPopup_EdgeProfile") {
            try ProcessClose(proc.ProcessId)
        }
    }
}

DestroyIslands() {
    global LeftIslandGui, RightIslandGui
    if IsObject(LeftIslandGui) {
        try LeftIslandGui.Destroy()
        LeftIslandGui := ""
    }
    if IsObject(RightIslandGui) {
        try RightIslandGui.Destroy()
        RightIslandGui := ""
    }
}


; ==============================================================================
; 設定画面（Ctrl+S で UAC 昇格起動された設定専用プロセスでのみ実行）
; ==============================================================================
OpenSettingsWithUAC() {
    try {
        if (A_IsCompiled) {
            Run('*RunAs "' A_ScriptFullPath '" /settings')
        } else {
            Run('*RunAs "' A_AhkPath '" "' A_ScriptFullPath '" /settings')
        }
    } catch {
        ; UACがキャンセルされた場合は何もしない
    }
}

ShowSettingsWindow() {
    global Config

    settingsGui := Gui("+AlwaysOnTop", "ShopActivityPopup 設定（管理者モード）")
    settingsGui.SetFont("s10", "Yu Gothic UI")
    settingsGui.MarginX := 20
    settingsGui.MarginY := 16

    settingsGui.Add("Text", , "表示する Web サイトの URL (http:// または https://):")
    edtUrl := settingsGui.Add("Edit", "w420", Config.TargetUrl)

    settingsGui.Add("Text", "y+14", "ポップアップ幅 (px) [0 = 作業領域の80%自動 / 指定時は320以上]:")
    edtWidth := settingsGui.Add("Edit", "w140 Number", Config.PopupWidth)

    settingsGui.Add("Text", "y+12", "ポップアップ高さ (px) [0 = 作業領域の80%自動 / 指定時は240以上]:")
    edtHeight := settingsGui.Add("Edit", "w140 Number", Config.PopupHeight)

    settingsGui.Add("Text", "y+12", "無操作で自動終了するまでの秒数 [1秒以上]:")
    edtIdle := settingsGui.Add("Edit", "w140 Number", Config.CloseAfterIdleSeconds)

    btnSave := settingsGui.Add("Button", "y+20 w100 h32 Default", "保存")
    btnCancel := settingsGui.Add("Button", "x+12 w100 h32", "キャンセル")

    btnSave.OnEvent("Click", (*) => ValidateAndSave(settingsGui, edtUrl.Value, edtWidth.Value, edtHeight.Value, edtIdle.Value))
    btnCancel.OnEvent("Click", (*) => ExitApp())
    settingsGui.OnEvent("Close", (*) => ExitApp())

    settingsGui.Show()
}

ValidateAndSave(guiObj, rawUrl, rawW, rawH, rawIdle) {
    url := Trim(rawUrl)
    w := (rawW = "") ? -1 : Integer(rawW)
    h := (rawH = "") ? -1 : Integer(rawH)
    idle := (rawIdle = "") ? 0 : Integer(rawIdle)

    ; 仕様書14.3 バリデーション
    if !RegExMatch(url, "i)^https?://[^\s/$.?#].[^\s]*$") {
        MsgBox("URL は http:// または https:// で始まる有効な絶対 URL を入力してください。", "入力エラー", "Icon!")
        return
    }
    if (w != 0 && w < 320) {
        MsgBox("幅は 0（自動80%）または 320px 以上を指定してください。", "入力エラー", "Icon!")
        return
    }
    if (h != 0 && h < 240) {
        MsgBox("高さは 0（自動80%）または 240px 以上を指定してください。", "入力エラー", "Icon!")
        return
    }
    if (idle < 1) {
        MsgBox("無操作秒数は 1 秒以上を指定してください。", "入力エラー", "Icon!")
        return
    }

    newCfg := {
        TargetUrl: url,
        PopupWidth: w,
        PopupHeight: h,
        CloseAfterIdleSeconds: idle
    }

    try {
        SaveConfig(newCfg)
        guiObj.Destroy()
        ExitApp()
    } catch as err {
        MsgBox("設定ファイルの保存に失敗しました。`n" err.Message, "保存エラー", "Iconx")
    }
}


; ==============================================================================
; appsettings.json 読み書き（外部ライブラリ不要のJSONパーサー）
; ==============================================================================
LoadConfig() {
    global Config, ConfigFile
    if !FileExist(ConfigFile)
        return

    jsonText := FileRead(ConfigFile, "UTF-8")

    if RegExMatch(jsonText, '"TargetUrl"\s*:\s*"((?:\\.|[^"\\])*)"', &mUrl) {
        Config.TargetUrl := StrReplace(StrReplace(mUrl[1], '\"', '"'), "\\", "\")
    }
    if RegExMatch(jsonText, '"PopupWidth"\s*:\s*(\d+)', &mW) {
        Config.PopupWidth := Integer(mW[1])
    }
    if RegExMatch(jsonText, '"PopupHeight"\s*:\s*(\d+)', &mH) {
        Config.PopupHeight := Integer(mH[1])
    }
    if RegExMatch(jsonText, '"CloseAfterIdleSeconds"\s*:\s*(\d+)', &mIdle) {
        Config.CloseAfterIdleSeconds := Integer(mIdle[1])
    }
}

SaveConfig(cfg) {
    global ConfigFile
    escapedUrl := StrReplace(StrReplace(cfg.TargetUrl, "\", "\\"), '"', '\"')
    json := Format(
        '{{`r`n'
        '  "TargetUrl": "{1}",`r`n'
        '  "PopupWidth": {2},`r`n'
        '  "PopupHeight": {3},`r`n'
        '  "CloseAfterIdleSeconds": {4}`r`n'
        '}}',
        escapedUrl, cfg.PopupWidth, cfg.PopupHeight, cfg.CloseAfterIdleSeconds
    )
    if FileExist(ConfigFile)
        FileDelete(ConfigFile)
    FileAppend(json, ConfigFile, "UTF-8-RAW")
}


; ==============================================================================
; Win32 API ヘルパー関数（丸角適用・オーナー設定・Edge検出・多重起動防止）
; ==============================================================================
ApplyRoundCorners(hwnd, width, height, radius := 16) {
    ; Windows 11 DWM 丸角属性 (DWMWA_WINDOW_CORNER_PREFERENCE = 33, DWMWCP_ROUND = 2)
    try DllCall("dwmapi\DwmSetWindowAttribute", "Ptr", hwnd, "Int", 33, "Int*", 2, "Int", 4)
    ; Windows 10 / 11 両対応 GDI リージョン丸角クリッピング
    hRgn := DllCall("gdi32\CreateRoundRectRgn", "Int", 0, "Int", 0, "Int", width + 1, "Int", height + 1, "Int", radius, "Int", radius, "Ptr")
    if (hRgn) {
        DllCall("user32\SetWindowRgn", "Ptr", hwnd, "Ptr", hRgn, "Int", true)
    }
}

SetWindowOwner(childHwnd, ownerHwnd) {
    ; GWLP_HWNDPARENT = -8
    DllCall("user32\SetWindowLongPtr", "Ptr", childHwnd, "Int", -8, "Ptr", ownerHwnd, "Ptr")
}

FindEdgePath() {
    paths := [
        "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        "C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        EnvGet("LocalAppData") "\Microsoft\Edge\Application\msedge.exe"
    ]
    for p in paths {
        if FileExist(p)
            return p
    }
    return ""
}

EnforceSingleResidentInstance() {
    currentPid := ProcessExist()
    scriptName := A_ScriptName
    for proc in ComObjGet("winmgmts:").ExecQuery("SELECT ProcessId, CommandLine FROM Win32_Process") {
        if (proc.ProcessId != currentPid && InStr(proc.CommandLine, scriptName) && !InStr(proc.CommandLine, "/settings")) {
            ExitApp()
        }
    }
}
