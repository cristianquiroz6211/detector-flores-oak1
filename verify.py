#!/usr/bin/env python3
"""
🔍 VERIFICADOR DEL SISTEMA - Detector de Flores OAK-1
====================================================

Script de verificación que valida la instalación completa
y muestra información detallada del sistema.

Uso:
    python verify.py

Autor: Cristian Quiroz
"""

import sys
import os
import platform
from pathlib import Path
import subprocess
import time

# Colores para output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header():
    """Muestra el header del verificador"""
    print(f"{Colors.CYAN}{Colors.BOLD}")
    print("=" * 60)
    print("🔍 VERIFICADOR DEL SISTEMA - Detector de Flores OAK-1")
    print("=" * 60)
    print(f"{Colors.END}")
    print(f"{Colors.BLUE}Validando instalación y configuración del sistema{Colors.END}")
    print()

def check_python():
    """Verifica Python y versión"""
    print(f"{Colors.YELLOW}🐍 Verificando Python...{Colors.END}")
    
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    if version >= (3, 8):
        print(f"{Colors.GREEN}✅ Python {version_str} - Compatible{Colors.END}")
        return True
    else:
        print(f"{Colors.RED}❌ Python {version_str} - Se requiere 3.8+{Colors.END}")
        return False

def check_system_info():
    """Muestra información del sistema"""
    print(f"{Colors.YELLOW}💻 Información del Sistema...{Colors.END}")
    
    system = platform.system()
    release = platform.release()
    arch = platform.machine()
    processor = platform.processor()
    
    print(f"{Colors.GREEN}✅ SO: {system} {release} ({arch}){Colors.END}")
    if processor:
        print(f"{Colors.BLUE}ℹ️  Procesador: {processor}{Colors.END}")
    
    return True

def check_project_files():
    """Verifica archivos del proyecto"""
    print(f"{Colors.YELLOW}📁 Verificando archivos del proyecto...{Colors.END}")
    
    required_files = {
        'detector_flores.py': 'Script principal del detector',
        'best.blob': 'Modelo YOLO para OAK-1',
        'requirements.txt': 'Dependencias de Python',
        'README.md': 'Documentación principal',
        'LICENSE': 'Licencia del proyecto'
    }
    
    optional_files = {
        'CHANGELOG.md': 'Historial de versiones',
        'setup.py': 'Configuración de instalación',
        'install.py': 'Instalador automático',
        '.gitignore': 'Exclusiones de Git'
    }
    
    all_good = True
    
    # Archivos requeridos
    for filename, description in required_files.items():
        path = Path(filename)
        if path.exists():
            size = path.stat().st_size
            size_str = f"({size:,} bytes)" if size < 1024*1024 else f"({size/(1024*1024):.1f} MB)"
            print(f"{Colors.GREEN}✅ {filename} - {description} {size_str}{Colors.END}")
        else:
            print(f"{Colors.RED}❌ {filename} - {description} - NO ENCONTRADO{Colors.END}")
            all_good = False
    
    # Archivos opcionales
    for filename, description in optional_files.items():
        path = Path(filename)
        if path.exists():
            print(f"{Colors.GREEN}✅ {filename} - {description}{Colors.END}")
    
    return all_good

def check_virtual_env():
    """Verifica entorno virtual"""
    print(f"{Colors.YELLOW}🔧 Verificando entorno virtual...{Colors.END}")
    
    venv_path = Path(".venv")
    if venv_path.exists():
        print(f"{Colors.GREEN}✅ Entorno virtual encontrado{Colors.END}")
        
        # Verificar activación
        if sys.prefix != sys.base_prefix:
            print(f"{Colors.GREEN}✅ Entorno virtual está ACTIVO{Colors.END}")
        else:
            print(f"{Colors.YELLOW}⚠️  Entorno virtual NO está activo{Colors.END}")
            print(f"{Colors.YELLOW}   Activa con: .venv\\Scripts\\activate (Windows){Colors.END}")
        
        return True
    else:
        print(f"{Colors.RED}❌ Entorno virtual no encontrado{Colors.END}")
        print(f"{Colors.YELLOW}   Crea con: python -m venv .venv{Colors.END}")
        return False

def check_dependencies():
    """Verifica dependencias instaladas"""
    print(f"{Colors.YELLOW}📦 Verificando dependencias...{Colors.END}")
    
    dependencies = [
        ('depthai', 'DepthAI Framework'),
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
        ('PIL', 'Pillow')
    ]
    
    all_installed = True
    
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"{Colors.GREEN}✅ {name} - Instalado{Colors.END}")
        except ImportError:
            print(f"{Colors.RED}❌ {name} - NO instalado{Colors.END}")
            all_installed = False
    
    return all_installed

def check_oak_devices():
    """Verifica dispositivos OAK conectados"""
    print(f"{Colors.YELLOW}📹 Verificando dispositivos OAK-1...{Colors.END}")
    
    try:
        import depthai as dai
        devices = dai.Device.getAllAvailableDevices()
        
        if devices:
            print(f"{Colors.GREEN}✅ {len(devices)} dispositivo(s) OAK encontrado(s):{Colors.END}")
            for i, device in enumerate(devices):
                print(f"{Colors.BLUE}   [{i+1}] {device.getMxId()} - {device.name}{Colors.END}")
            return True
        else:
            print(f"{Colors.YELLOW}⚠️  No se encontraron dispositivos OAK-1{Colors.END}")
            print(f"{Colors.YELLOW}   Verifica que la cámara esté conectada{Colors.END}")
            return False
            
    except ImportError:
        print(f"{Colors.RED}❌ DepthAI no instalado - No se pueden verificar dispositivos{Colors.END}")
        return False

def check_git_status():
    """Verifica estado de Git"""
    print(f"{Colors.YELLOW}📝 Verificando repositorio Git...{Colors.END}")
    
    git_dir = Path(".git")
    if git_dir.exists():
        try:
            # Verificar status
            result = subprocess.run(['git', 'status', '--porcelain'], 
                                  capture_output=True, text=True, check=True)
            
            if result.stdout.strip():
                print(f"{Colors.YELLOW}⚠️  Hay cambios sin commitear{Colors.END}")
            else:
                print(f"{Colors.GREEN}✅ Repositorio limpio{Colors.END}")
            
            # Verificar remoto
            result = subprocess.run(['git', 'remote', 'get-url', 'origin'], 
                                  capture_output=True, text=True, check=True)
            remote_url = result.stdout.strip()
            print(f"{Colors.BLUE}ℹ️  Remoto: {remote_url}{Colors.END}")
            
            return True
            
        except subprocess.CalledProcessError:
            print(f"{Colors.YELLOW}⚠️  Error verificando Git{Colors.END}")
            return False
    else:
        print(f"{Colors.YELLOW}⚠️  No es un repositorio Git{Colors.END}")
        return False

def run_syntax_check():
    """Verifica sintaxis del código principal"""
    print(f"{Colors.YELLOW}🔍 Verificando sintaxis del código...{Colors.END}")
    
    try:
        import detector_flores
        print(f"{Colors.GREEN}✅ detector_flores.py - Sintaxis correcta{Colors.END}")
        return True
    except SyntaxError as e:
        print(f"{Colors.RED}❌ Error de sintaxis en detector_flores.py: {e}{Colors.END}")
        return False
    except ImportError as e:
        print(f"{Colors.YELLOW}⚠️  Error de import (dependencias): {e}{Colors.END}")
        return False
    except Exception as e:
        print(f"{Colors.RED}❌ Error inesperado: {e}{Colors.END}")
        return False

def show_summary(results):
    """Muestra resumen de resultados"""
    print(f"\n{Colors.CYAN}{Colors.BOLD}📊 RESUMEN DE VERIFICACIÓN{Colors.END}")
    print(f"{Colors.CYAN}=" * 50 + f"{Colors.END}")
    
    total_checks = len(results)
    passed_checks = sum(results.values())
    failed_checks = total_checks - passed_checks
    
    print(f"\n{Colors.BOLD}Resultados:{Colors.END}")
    print(f"{Colors.GREEN}✅ Verificaciones exitosas: {passed_checks}{Colors.END}")
    print(f"{Colors.RED}❌ Verificaciones fallidas: {failed_checks}{Colors.END}")
    print(f"{Colors.BLUE}📊 Total de verificaciones: {total_checks}{Colors.END}")
    
    # Calcular porcentaje
    percentage = (passed_checks / total_checks) * 100
    
    if percentage == 100:
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 ¡SISTEMA COMPLETAMENTE VERIFICADO! ({percentage:.0f}%){Colors.END}")
        print(f"{Colors.GREEN}El detector está listo para usarse.{Colors.END}")
    elif percentage >= 80:
        print(f"\n{Colors.YELLOW}{Colors.BOLD}⚠️  SISTEMA MAYORMENTE FUNCIONAL ({percentage:.0f}%){Colors.END}")
        print(f"{Colors.YELLOW}Algunas funciones podrían no funcionar completamente.{Colors.END}")
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}🚨 REQUIERE ATENCIÓN ({percentage:.0f}%){Colors.END}")
        print(f"{Colors.RED}El sistema necesita configuración adicional.{Colors.END}")
    
    # Mostrar próximos pasos
    print(f"\n{Colors.BOLD}📋 Próximos pasos:{Colors.END}")
    
    if results.get('dependencies', False) and results.get('files', False):
        print(f"1. {Colors.CYAN}python detector_flores.py{Colors.END} - Ejecutar el detector")
        if not results.get('oak_devices', False):
            print(f"   {Colors.YELLOW}(Conecta la cámara OAK-1 primero){Colors.END}")
    else:
        print(f"1. {Colors.CYAN}python install.py{Colors.END} - Ejecutar instalador automático")
    
    print(f"2. {Colors.CYAN}README.md{Colors.END} - Consultar documentación completa")
    print(f"3. {Colors.CYAN}https://github.com/cristianquiroz6211/detector-flores-oak1{Colors.END} - Ver repositorio")

def main():
    """Función principal"""
    print_header()
    
    # Ejecutar verificaciones
    results = {}
    
    results['python'] = check_python()
    results['system'] = check_system_info()
    results['files'] = check_project_files()
    results['venv'] = check_virtual_env()
    results['dependencies'] = check_dependencies()
    results['oak_devices'] = check_oak_devices()
    results['git'] = check_git_status()
    results['syntax'] = run_syntax_check()
    
    # Mostrar resumen
    show_summary(results)

if __name__ == "__main__":
    main()
