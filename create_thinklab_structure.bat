@echo off
setlocal enabledelayedexpansion

echo Creating ThinkLab folder structure for Windows...
echo.

:: Set the base directory
set BASE_DIR=%CD%\thinklab

:: Create main directory
if not exist "%BASE_DIR%" mkdir "%BASE_DIR%"
cd "%BASE_DIR%"

:: Create root level files (empty)
type nul > README.md
type nul > setup.py
type nul > requirements.txt
type nul > LICENSE
type nul > .gitignore
type nul > docker-compose.yml

:: Create configs directory and subdirectories
mkdir configs 2>nul
mkdir configs\models 2>nul
mkdir configs\server 2>nul
mkdir configs\research 2>nul

:: Create thinklab package directory
mkdir thinklab 2>nul
cd thinklab

:: Create __init__.py
type nul > __init__.py

:: Create core directory
mkdir core 2>nul
cd core
type nul > __init__.py
type nul > base_model.py
cd ..

:: Create models directory and subdirectories
mkdir models 2>nul
cd models
type nul > __init__.py

:: llm subdirectory
mkdir llm 2>nul
cd llm
type nul > __init__.py
type nul > vanilla_gpt.py
type nul > qwen.py
type nul > llama2.py
type nul > llama3.py
type nul > llama4.py
type nul > gamma_models.py
cd ..

:: vision subdirectory
mkdir vision 2>nul
cd vision
type nul > __init__.py

:: sam subdirectory
mkdir sam 2>nul
cd sam
type nul > __init__.py
type nul > sam_model.py
cd ..

:: diffusion subdirectory
mkdir diffusion 2>nul
cd diffusion
type nul > __init__.py
type nul > diffusion.py
cd ..
cd ..

:: multimodal subdirectory
mkdir multimodal 2>nul
cd multimodal
type nul > __init__.py
type nul > gamma_multimodal_arch_vit.py
cd ..

:: ModelExplain subdirectory
mkdir ModelExplain 2>nul
cd ModelExplain
type nul > __init__.py
type nul > Grad-CAM.py
type nul > LIME.py
cd ..
cd ..

:: Create weights directory
mkdir weights 2>nul
cd weights
type nul > __init__.py
type nul > huggingface.py
cd ..

:: Create server directory and subdirectories
mkdir server 2>nul
cd server
type nul > __init__.py

:: models subdirectory (MVC)
mkdir models 2>nul
cd models
type nul > __init__.py
type nul > research_session.py
type nul > experiment_data.py
type nul > model_metadata.py
type nul > user_preferences.py
cd ..

:: views subdirectory
mkdir views 2>nul
cd views
type nul > __init__.py
type nul > api_endpoints.py
type nul > websocket_handler.py
type nul > response_formatter.py
type nul > error_handler.py
cd ..

:: controllers subdirectory
mkdir controllers 2>nul
cd controllers
type nul > __init__.py
type nul > research_controller.py
type nul > model_controller.py
type nul > multimodal_controller.py
type nul > analysis_controller.py
cd ..

:: chatbot subdirectory
mkdir chatbot 2>nul
cd chatbot
type nul > __init__.py
cd ..

:: middleware subdirectory
mkdir middleware 2>nul
cd middleware
type nul > __init__.py
type nul > authentication.py
type nul > rate_limiting.py
type nul > logging.py
type nul > caching.py
cd ..
cd ..

:: Create training directory and subdirectories
mkdir training 2>nul
cd training
type nul > __init__.py

:: datasets subdirectory
mkdir datasets 2>nul
cd datasets
type nul > __init__.py
type nul > multimodal_loader.py
type nul > benchmark_datasets.py
type nul > synthetic_generator.py
cd ..

:: distributed subdirectory
mkdir distributed 2>nul
cd distributed
type nul > __init__.py
type nul > data_parallel.py
type nul > model_parallel.py
type nul > federated.py
cd ..

:: optimization subdirectory
mkdir optimization 2>nul
cd optimization
type nul > __init__.py
type nul > adamw_variants.py
type nul > learning_schedules.py
type nul > gradient_analysis.py
cd ..

:: losses subdirectory
mkdir losses 2>nul
cd losses
type nul > __init__.py
cd ..
cd ..

echo.
echo ThinkLab folder structure created successfully at: %BASE_DIR%
echo.
tree %BASE_DIR% /F
pause