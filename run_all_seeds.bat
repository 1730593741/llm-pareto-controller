@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

REM ==========================================
REM 配置区
REM ==========================================
set SEEDS=42 100 2024 1234 9999

REM 你的 Python 命令，通常直接 python 即可
set PYTHON_CMD=python

REM 临时配置文件输出目录
set TEMP_CFG_DIR=tmp_seed_configs

REM 结果输出根目录
set RUNS_ROOT=experiments\runs\main_results

REM 基础配置文件
set CFG_BASELINE=experiments\configs\baseline_nsga2.yaml
set CFG_RULE=experiments\configs\rule_control.yaml
set CFG_MOCK=experiments\configs\mock_llm.yaml
set CFG_REAL=experiments\configs\real_llm.yaml

if not exist "%TEMP_CFG_DIR%" mkdir "%TEMP_CFG_DIR%"
if not exist "%RUNS_ROOT%" mkdir "%RUNS_ROOT%"

echo ==========================================
echo [INFO] 开始批量运行多 seed 主结果实验
echo [INFO] Seeds: %SEEDS%
echo ==========================================

for %%s in (%SEEDS%) do (
    echo.
    echo ==========================================
    echo [INFO] 正在运行 Seed: %%s
    echo ==========================================

    REM ------------------------------------------
    REM 1/4 Baseline NSGA-II
    REM ------------------------------------------
    set BASE_OUT=%RUNS_ROOT%\baseline_nsga2\seed_%%s
    set BASE_CFG=%TEMP_CFG_DIR%\baseline_nsga2_seed_%%s.yaml

    echo [1/4] 生成 Baseline 配置...
    %PYTHON_CMD% make_seed_config.py "%CFG_BASELINE%" %%s "!BASE_OUT!" "!BASE_CFG!"
    if errorlevel 1 (
        echo [ERROR] Baseline 配置生成失败，Seed=%%s
        exit /b 1
    )

    echo [1/4] 运行 Baseline NSGA-II...
    %PYTHON_CMD% -c "from experiments.baselines.runner import run_baseline_nsga2; print(run_baseline_nsga2(r'!BASE_CFG!'))"
    if errorlevel 1 (
        echo [ERROR] Baseline NSGA-II 运行失败，Seed=%%s
        exit /b 1
    )

    REM ------------------------------------------
    REM 2/4 Rule Control
    REM ------------------------------------------
    set RULE_OUT=%RUNS_ROOT%\rule_control\seed_%%s
    set RULE_CFG=%TEMP_CFG_DIR%\rule_control_seed_%%s.yaml

    echo [2/4] 生成 Rule Control 配置...
    %PYTHON_CMD% make_seed_config.py "%CFG_RULE%" %%s "!RULE_OUT!" "!RULE_CFG!"
    if errorlevel 1 (
        echo [ERROR] Rule Control 配置生成失败，Seed=%%s
        exit /b 1
    )

    echo [2/4] 运行 Rule Control...
    %PYTHON_CMD% -c "from experiments.baselines.runner import run_rule_control_baseline; print(run_rule_control_baseline(r'!RULE_CFG!'))"
    if errorlevel 1 (
        echo [ERROR] Rule Control 运行失败，Seed=%%s
        exit /b 1
    )

    REM ------------------------------------------
    REM 3/4 Mock LLM
    REM ------------------------------------------
    set MOCK_OUT=%RUNS_ROOT%\mock_llm\seed_%%s
    set MOCK_CFG=%TEMP_CFG_DIR%\mock_llm_seed_%%s.yaml

    echo [3/4] 生成 Mock LLM 配置...
    %PYTHON_CMD% make_seed_config.py "%CFG_MOCK%" %%s "!MOCK_OUT!" "!MOCK_CFG!"
    if errorlevel 1 (
        echo [ERROR] Mock LLM 配置生成失败，Seed=%%s
        exit /b 1
    )

    echo [3/4] 运行 Mock LLM...
    %PYTHON_CMD% -c "from main import main; main(r'!MOCK_CFG!')"
    if errorlevel 1 (
        echo [ERROR] Mock LLM 运行失败，Seed=%%s
        exit /b 1
    )

    REM ------------------------------------------
    REM 4/4 Real LLM
    REM ------------------------------------------
    set REAL_OUT=%RUNS_ROOT%\real_llm\seed_%%s
    set REAL_CFG=%TEMP_CFG_DIR%\real_llm_seed_%%s.yaml

    echo [4/4] 生成 Real LLM 配置...
    %PYTHON_CMD% make_seed_config.py "%CFG_REAL%" %%s "!REAL_OUT!" "!REAL_CFG!"
    if errorlevel 1 (
        echo [ERROR] Real LLM 配置生成失败，Seed=%%s
        exit /b 1
    )

    echo [4/4] 运行 Real LLM...
    %PYTHON_CMD% -c "from main import main; main(r'!REAL_CFG!')"
    if errorlevel 1 (
        echo [ERROR] Real LLM 运行失败，Seed=%%s
        exit /b 1
    )

    echo [INFO] Seed %%s 运行完成
    echo [INFO] 结果目录:
    echo        !BASE_OUT!
    echo        !RULE_OUT!
    echo        !MOCK_OUT!
    echo        !REAL_OUT!
)

echo.
echo ==========================================
echo [SUCCESS] 所有实验运行完毕！
echo [INFO] 请查看 summary.json:
echo        %RUNS_ROOT%\baseline_nsga2\seed_*\summary.json
echo        %RUNS_ROOT%\rule_control\seed_*\summary.json
echo        %RUNS_ROOT%\mock_llm\seed_*\summary.json
echo        %RUNS_ROOT%\real_llm\seed_*\summary.json
echo ==========================================
pause