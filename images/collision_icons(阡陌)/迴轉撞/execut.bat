::迴轉撞分類參考
https://www.perplexity.ai/search/zhe-xie-shi-hui-zhuan-zhuang-d-aav38xQRSFW8dB0R98hEIQ#0

@echo off
chcp 65001
setlocal

call :CopyImages "路口中端" "左彎迴轉撞"
call :CopyImages "西側鄰近路口" "右轉迴轉撞"
call :CopyImages "西側路口近端" "右轉迴轉撞 左彎迴轉撞"
call :CopyImages "東側路口遠端" " 左彎迴轉撞"
call :CopyImages "東側鄰近路口" "右轉迴轉撞"
call :CopyImages "東側路口近端" "右轉迴轉撞 左彎迴轉撞"
call :CopyImages "西側路口遠端" "左彎迴轉撞"
call :CopyImages "南側鄰近路口" " 右轉迴轉撞"
call :CopyImages "南側路口近端" "右轉迴轉撞 左彎迴轉撞"
call :CopyImages "北側路口遠端" " 左彎迴轉撞"
call :CopyImages "北側鄰近路口" "右轉迴轉撞"
call :CopyImages "北側路口近端" "右轉迴轉撞 左彎迴轉撞"
call :CopyImages "南側路口遠端" "左彎迴轉撞"

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

REM === Function ===
::REM 定義複製指定副檔名圖片的函式：參數1=目標資料夾名稱，參數2=副檔名條件
:CopySubFormatImages
    set "TargetDir=%~1"
    set "TypeList=%~2"

    echo 目標資料夾：%TargetDir%
    if not exist "%TargetDir%" (
        md "%TargetDir%"
    )

    for %%i in (%TypeList%) do (
        if exist "%%i" (
            echo 正在複製 %%i 至 "%TargetDir%"
            copy "%%i" "%TargetDir%\" >nul
        ) else (
            echo [警告] 找不到圖片檔：%%i
        )
    )
    echo 此批類型複製完成。
    echo.
    goto :eof    


echo 全部分類複製完成！ 
pause
