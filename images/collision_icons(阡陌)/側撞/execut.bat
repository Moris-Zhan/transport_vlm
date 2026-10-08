::側撞的分類類別參考
https://www.perplexity.ai/search/zhe-xie-shi-ce-zhuang-de-fen-l-moSn4HtNTlOPbSTDQV14DA#0

@echo off
chcp 65001
setlocal

REM === 主程式 ===
call :CopyImages "路口中端" "左彎側撞 左彎側撞_1 左彎側撞_2 左彎側撞_3 右彎側撞 右彎側撞_1"
call :CopyImages "西側路段(往東)" "左後側撞 左後側撞_1 右後側撞 右後側撞_2"
call :CopyImages "西側鄰近路口" "左彎側撞 左彎側撞_1 左彎側撞_2 左彎側撞_3 右彎側撞 右彎側撞_1"
call :CopyImages "西側路口近端" "左彎側撞 左彎側撞_1 左彎側撞_2 左彎側撞_3 左後側撞 左後側撞_1"
call :CopyImages "東側路口遠端" "右側撞 右側撞_1 左側撞 左側撞_1"
call :CopyImages "東側路段(往東)" "左後側撞 左後側撞_1 右後側撞 右後側撞_2"
call :CopyImages "東側路段(往西)" "左後側撞 左後側撞_1 右後側撞 右後側撞_2"
call :CopyImages "東側鄰近路口" "左彎側撞 左彎側撞_1 左彎側撞_2 左彎側撞_3 右彎側撞 右彎側撞_1"
call :CopyImages "東側路口近端" "左彎側撞 左彎側撞_1 左彎側撞_2 左彎側撞_3 右彎側撞 右彎側撞_1"
call :CopyImages "西側路口遠端" "左側撞 左側撞_1 右側撞 右側撞_1"
call :CopyImages "西側路段(往西)" "左後側撞 左後側撞_1 右後側撞 右後側撞_2"
call :CopyImages "南側路段(往北)" "左後側撞 左後側撞_1 右後側撞 右後側撞_2"
call :CopyImages "南側鄰近路口" "左彎側撞 左彎側撞_1 左彎側撞_2 左彎側撞_3 右側撞 右側撞_1"
call :CopyImages "南側路口近端" "右彎側撞 右彎側撞_1 右側撞 右側撞_1"
call :CopyImages "南側路口遠端" "左側撞 左側撞_1 右側撞 右側撞_1"
call :CopyImages "北側路口遠端" "左側撞 左側撞_1 右側撞 右側撞_1"
call :CopyImages "北側路段(往北)" "左後側撞 左後側撞_1 右後側撞 右後側撞_2"
call :CopyImages "北側路段(往南)" "左後側撞 左後側撞_1 右後側撞 右後側撞_2"
call :CopyImages "北側鄰近路口" "左彎側撞 左彎側撞_1 左彎側撞_2 左彎側撞_3 右彎側撞 右彎側撞_1"
call :CopyImages "北側路口近端" "左彎側撞 左彎側撞_1 左彎側撞_2 左彎側撞_3 左後側撞 左後側撞_1"
call :CopyImages "南側路段(往南)" "左後側撞 左後側撞_1 右後側撞 右後側撞_2"
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