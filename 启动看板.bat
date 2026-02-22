@echo off
chcp 65001 >nul
title 中证50期权策略看板

echo ========================================
echo   中证50期权策略看板
echo ========================================
echo.

REM 获取Python路径
set PYTHON=C:\Users\gaaiy\AppData\Local\Programs\Python\Python312\python.exe
set STREAMLIT=C:\Users\gaaiy\AppData\Local\Programs\Python\Python312\Scripts\streamlit.exe

REM 检查Python
"%PYTHON%" --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到Python，请先安装Python 3.12+
    pause
    exit /b 1
)

echo [1/3] 检查依赖包...
"%PYTHON%" -c "import streamlit" 2>nul
if errorlevel 1 (
    echo [警告] 正在安装依赖包...
    "%PYTHON%" -m pip install streamlit akshare pandas numpy arch statsmodels -q
)

echo [2/3] 启动看板...
echo.
echo 请在浏览器中打开: http://localhost:8501
echo.
echo 按 Ctrl+C 可停止服务
echo.

REM 启动Streamlit (静默模式)
"%STREAMLIT%" run "%~dp0app.py" --server.headless=true --browser.gatherUsageStats=false

pause
