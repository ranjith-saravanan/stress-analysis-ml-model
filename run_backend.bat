@echo off
echo ========================================
echo  Social Media Detox Effect Analyzer
echo  Flask REST API Backend
echo ========================================
echo.
echo Starting the backend API server...
echo.
echo The API will be available at:
echo http://localhost:5000
echo.
echo API Endpoints:
echo   GET  /
echo   POST /api/predict
echo   GET  /api/stats
echo   GET  /api/correlations
echo   GET  /api/clusters
echo   POST /api/recommendations
echo   GET  /api/export
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

python backend_api.py
