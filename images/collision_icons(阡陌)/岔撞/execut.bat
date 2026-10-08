::岔撞的分類類別參考
https://www.perplexity.ai/search/zhe-xie-shi-cha-zhuang-de-fen-.kRjFre.RAujSdkh4Ejhog#0

@echo off
chcp 65001
setlocal

call :CopyImages "路口中端" "左轉岔撞 右轉岔撞"
call :CopyImages "西側路口近端" "右轉岔撞_1 右岔撞"
call :CopyImages "西側路口遠端" "左後岔撞 右後岔撞"
call :CopyImages "西側鄰近路口" "左前岔撞 右前岔撞"
call :CopyImages "西側路段(往東)" "左後岔撞 右後岔撞"
call :CopyImages "西側路段(往西)" "左後岔撞 右後岔撞"
call :CopyImages "南側路口近端" "左轉岔撞_1 左轉岔撞"
call :CopyImages "南側路口遠端" "左後岔撞 右後岔撞"
call :CopyImages "南側鄰近路口" "左前岔撞 右前岔撞"
call :CopyImages "南側路段(往北)" "左後岔撞 右後岔撞"
call :CopyImages "南側路段(往南)" "左後岔撞 右後岔撞"
call :CopyImages "北側路口近端" "右轉岔撞 右轉岔撞_1"
call :CopyImages "北側路口遠端" "左後岔撞 右後岔撞"
call :CopyImages "北側鄰近路口" "左前岔撞 右前岔撞"
call :CopyImages "北側路段(往北)" "左後岔撞 右後岔撞"
call :CopyImages "北側路段(往南)" "左後岔撞 右後岔撞"
call :CopyImages "東側路口近端" "左轉岔撞 右轉岔撞_1"
call :CopyImages "東側路口遠端" "左後岔撞 右後岔撞"
call :CopyImages "東側鄰近路口" "左前岔撞 右前岔撞"
call :CopyImages "東側路段(往東)" "左後岔撞 右後岔撞"
call :CopyImages "東側路段(往西)" "左後岔撞 右後岔撞"

echo 全部分類複製完成！
pause


REM === Function ===
REM 定義複製圖片的函式：參數1=目標資料夾名稱，參數2=類型清單
:CopyImages
    set "TargetDir=%~1"
    set "TypeList=%~2"

    echo 目標資料夾：%TargetDir%
    if not exist "%TargetDir%" (
        md "%TargetDir%"
    )

    for %%i in (%TypeList%) do (
        if exist "%%i.png" (
            echo 正在複製 %%i.png 至 "%TargetDir%"
            copy "%%i.png" "%TargetDir%\" >nul
        ) else (
            echo [警告] 找不到圖片檔：%%i.png
        )
    )
    echo 此批類型複製完成。
    echo.
    goto :eof


echo 全部分類複製完成！ 
pause
