@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

REM ==========================================
REM DWTA 真实难例（多波次连续扰动）多 seed 运行脚本
REM - 保持 env-driven LLM（不写任何真实 key）
REM - 输出目录兼容 collect_summaries.py
REM ==========================================
set SEEDS=42 100 2024 1234 9999
set PYTHON_CMD=python
set TEMP_CFG_DIR=tmp_seed_configs_dwta_hard
set RUNS_ROOT=experiments\runs\dwta_hard_results
set CFG_DWTA_BASE=experiments\configs\dwta_hard_realworld.yaml

if not exist "%TEMP_CFG_DIR%" mkdir "%TEMP_CFG_DIR%"
if not exist "%RUNS_ROOT%" mkdir "%RUNS_ROOT%"

echo ==========================================
echo [INFO] 开始批量运行 DWTA HARD REALWORLD 多 seed 实验
echo [INFO] Seeds: %SEEDS%
echo [INFO] Base config: %CFG_DWTA_BASE%
echo ==========================================

for %%s in (%SEEDS%) do (
    echo.
    echo ==========================================
    echo [INFO] 正在运行 Seed: %%s
    echo ==========================================

    set BASE_OUT=%RUNS_ROOT%\baseline_nsga2\seed_%%s
    set BASE_CFG=%TEMP_CFG_DIR%\baseline_nsga2_dwta_hard_seed_%%s.yaml
    echo [1/4] 生成 Baseline 配置...
    %PYTHON_CMD% make_dwta_run_config.py "%CFG_DWTA_BASE%" rule false %%s "!BASE_OUT!" "!BASE_CFG!"
    if errorlevel 1 exit /b 1
    echo [1/4] 运行 Baseline...
    %PYTHON_CMD% -c "from main import main; main(r'!BASE_CFG!')"
    if errorlevel 1 exit /b 1

    set RULE_OUT=%RUNS_ROOT%\rule_control\seed_%%s
    set RULE_CFG=%TEMP_CFG_DIR%\rule_control_dwta_hard_seed_%%s.yaml
    echo [2/4] 生成 Rule 配置...
    %PYTHON_CMD% make_dwta_run_config.py "%CFG_DWTA_BASE%" rule true %%s "!RULE_OUT!" "!RULE_CFG!"
    if errorlevel 1 exit /b 1
    echo [2/4] 运行 Rule Control...
    %PYTHON_CMD% -c "from main import main; main(r'!RULE_CFG!')"
    if errorlevel 1 exit /b 1

    set MOCK_OUT=%RUNS_ROOT%\mock_llm\seed_%%s
    set MOCK_CFG=%TEMP_CFG_DIR%\mock_llm_dwta_hard_seed_%%s.yaml
    echo [3/4] 生成 Mock 配置...
    %PYTHON_CMD% make_dwta_run_config.py "%CFG_DWTA_BASE%" mock_llm true %%s "!MOCK_OUT!" "!MOCK_CFG!"
    if errorlevel 1 exit /b 1
    echo [3/4] 运行 Mock LLM...
    %PYTHON_CMD% -c "from main import main; main(r'!MOCK_CFG!')"
    if errorlevel 1 exit /b 1

    set REAL_OUT=%RUNS_ROOT%\real_llm\seed_%%s
    set REAL_CFG=%TEMP_CFG_DIR%\real_llm_dwta_hard_seed_%%s.yaml
    echo [4/4] 生成 Real 配置...
    %PYTHON_CMD% make_dwta_run_config.py "%CFG_DWTA_BASE%" real_llm true %%s "!REAL_OUT!" "!REAL_CFG!"
    if errorlevel 1 exit /b 1
    echo [4/4] 运行 Real LLM...
    %PYTHON_CMD% -c "from main import main; main(r'!REAL_CFG!')"
    if errorlevel 1 exit /b 1
)

echo.
echo [SUCCESS] DWTA HARD REALWORLD 多 seed 实验运行完成。
echo [INFO] 汇总命令示例:
echo        python collect_summaries.py experiments/runs/dwta_hard_results experiments/exports/dwta_hard_results
pause
