@echo off
rem ========================================================================
rem Git Backup Automation Script (run03_backup.bat)
rem ========================================================================
rem 
rem DESCRIPTION:
rem     Interactive batch script for automated Git backup operations with
rem     version tagging and remote repository synchronization.
rem 
rem FEATURES:
rem     - Dynamic latest tag detection and display
rem     - Interactive version input with validation
rem     - Automated Git workflow (add, commit, push, tag)
rem     - Error handling and user feedback
rem     - Cross-platform compatible command structure
rem 
rem USAGE:
rem     run03_backup.bat
rem     
rem INPUT:
rem     Version string (e.g., v1.0.0, v2.1.3a, v3.0.0-beta)
rem     
rem OUTPUT:
rem     - Git commit with version message
rem     - Git tag with version annotation
rem     - Remote repository synchronization
rem     
rem REQUIREMENTS:
rem     - Git installed and configured
rem     - Valid Git repository
rem     - Remote origin configured
rem     - Write permissions to repository
rem     
rem AUTHOR:
rem     Created: 2025-08-10
rem     Modified: 2025-08-10
rem     Version: 1.0.0
rem 
rem LICENSE:
rem     MITO License - 2025 © Hideki Ono
rem 
rem ========================================================================

rem Interactive version input backup script

echo ========================================
echo        Git Backup Script
echo ========================================
echo.

rem Get latest tag as example (using git tag --sort=-version:refname)
for /f "tokens=1" %%i in ('git tag --sort^=-version:refname 2^>nul') do (
    set LATEST_TAG=%%i
    goto :found_tag
)
:found_tag
if "%LATEST_TAG%"=="" set LATEST_TAG=v0.0.0

rem Ask user to input version
set /p VERSION="Please enter version (ex: %LATEST_TAG%): "

rem Validate input
if "%VERSION%"=="" (
    echo Error: Version cannot be empty!
    pause
    exit /b 1
)

echo.
echo Selected version: %VERSION%
echo.

rem Execute git commands with user input version
echo Backup started: %VERSION%
git add . 
git commit -m "Backup version %VERSION%" 
git push origin main
git tag -a %VERSION% -m "Version %VERSION% backup" 
git push origin --tags
echo Backup completed: %VERSION%
echo.
pause