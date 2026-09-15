@echo off
rem ============================================================================
rem  movie-system - 启动后端（免 Maven 版，直接用已编译好的 jar）
rem
rem  适用场景：本机没有安装 Maven，但 backend\target 下已有可用的 jar 包。
rem  若本机装了 Maven，仍可按 README 使用 restart_backend.cmd。
rem
rem  前置条件：MySQL(3306) 与 Redis(6379) 必须先启动，否则后端启动失败。
rem  启动方式：双击本文件，日志写入项目根目录 _boot.log
rem  访问地址：http://localhost:8000/admin    账号 admin / 123456
rem ============================================================================
cd /d "%~dp0backend"
call :run > ..\_boot.log 2>&1
exit /b

:run
java -jar target\mediaAnalysisSystem-3.0.3.jar
