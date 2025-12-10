@echo off
chcp 65001 > nul
title Photoye - 本地智能照片管理工具

echo.
echo =====================================================
echo         Photoye - 本地智能照片管理工具
echo =====================================================
echo.

REM 检查Python是否安装
python --version > nul 2>&1
if errorlevel 1 (
    echo ❌ 错误: 未找到Python解释器
    echo.
    echo 请先安装Python 3.8或更高版本:
    echo https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

REM 显示Python版本
echo ✅ 检测到Python:
python --version

echo.
echo 正在启动Photoye...
echo.

REM 运行启动脚本
python run.py

REM 如果程序异常退出，显示错误信息
if errorlevel 1 (
    echo.
    echo ❌ 程序异常退出
    echo.
)

pause