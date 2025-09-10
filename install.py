#!/usr/bin/env python3
"""
🌸 INSTALADOR AUTOMÁTICO - DETECTOR DE FLORES OAK-1
==================================================

Script de instalación automática que configura el entorno
y verifica todas las dependencias necesarias.

Uso:
    python install.py

Autor: Cristian Quiroz
Proyecto: https://github.com/cristianquiroz6211/detector-flores-oak1
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

# Colores para output en terminal
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
    """Muestra el header del instalador"""
    print(f"{Colors.CYAN}{Colors.BOLD}")
    print("=" * 60)
    print("🌸 DETECTOR DE FLORES OAK-1 - INSTALADOR AUTOMÁTICO")
    print("=" * 60)
    print(f"{Colors.END}")
    print(f"{Colors.BLUE}Configurando entorno para detección de flores en tiempo real{Colors.END}")
    print(f"{Colors.BLUE}Autor: Cristian Quiroz{Colors.END}")
    print()

def check_python_version():
    """Verifica que Python sea compatible"""
    print(f"{Colors.YELLOW}🐍 Verificando versión de Python...{Colors.END}")
    
    version = sys.version_info
    min_version = (3, 8)
    
    if version >= min_version:
        print(f"{Colors.GREEN}✅ Python {version.major}.{version.minor}.{version.micro} OK{Colors.END}")
        return True
    else:
        print(f"{Colors.RED}❌ Python {version.major}.{version.minor} no es compatible{Colors.END}")
        print(f"{Colors.RED}   Se requiere Python {min_version[0]}.{min_version[1]}+{Colors.END}")
        return False

def check_system():
    """Verifica el sistema operativo"""
    print(f"{Colors.YELLOW}💻 Verificando sistema operativo...{Colors.END}")
    
    system = platform.system()
    print(f"{Colors.GREEN}✅ Sistema detectado: {system} {platform.release()}{Colors.END}")
    
    if system == "Windows":
        print(f"{Colors.BLUE}ℹ️  Asegúrate de tener Visual C++ Redistributables instalados{Colors.END}")
    
    return True

def run_command(command, description):
    """Ejecuta un comando y maneja errores"""
    print(f"{Colors.YELLOW}⚙️  {description}...{Colors.END}")
    
    try:
        # Ejecutar comando
        result = subprocess.run(
            command, 
            shell=True, 
            check=True, 
            capture_output=True, 
            text=True
        )
        print(f"{Colors.GREEN}✅ {description} completado{Colors.END}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"{Colors.RED}❌ Error en {description}{Colors.END}")
        print(f"{Colors.RED}   Comando: {command}{Colors.END}")
        print(f"{Colors.RED}   Error: {e.stderr.strip()}{Colors.END}")
        return False

def create_virtual_environment():
    """Crea el entorno virtual si no existe"""
    venv_path = Path(".venv")
    
    if venv_path.exists():
        print(f"{Colors.GREEN}✅ Entorno virtual ya existe{Colors.END}")
        return True
    
    print(f"{Colors.YELLOW}📦 Creando entorno virtual...{Colors.END}")
    return run_command("python -m venv .venv", "Creación de entorno virtual")

def activate_and_install():
    """Activa el entorno e instala dependencias"""
    system = platform.system()
    
    if system == "Windows":
        activate_cmd = ".venv\\Scripts\\activate"
        python_cmd = ".venv\\Scripts\\python"
        pip_cmd = ".venv\\Scripts\\pip"
    else:
        activate_cmd = "source .venv/bin/activate"
        python_cmd = ".venv/bin/python"
        pip_cmd = ".venv/bin/pip"
    
    # Actualizar pip
    if not run_command(f"{pip_cmd} install --upgrade pip", "Actualización de pip"):
        return False
    
    # Instalar wheel para compilaciones más rápidas
    run_command(f"{pip_cmd} install wheel", "Instalación de wheel (opcional)")
    
    # Instalar dependencias principales
    if not run_command(f"{pip_cmd} install -r requirements.txt", "Instalación de dependencias"):
        return False
    
    return True

def verify_installation():
    """Verifica que las dependencias estén correctamente instaladas"""
    print(f"{Colors.YELLOW}🔍 Verificando instalación...{Colors.END}")
    
    system = platform.system()
    python_cmd = ".venv\\Scripts\\python" if system == "Windows" else ".venv/bin/python"
    
    # Verificar importaciones críticas
    tests = [
        ("depthai", "DepthAI (OAK-1 support)"),
        ("cv2", "OpenCV (computer vision)"),
        ("numpy", "NumPy (numerical computing)"),
        ("PIL", "Pillow (image processing)")
    ]
    
    all_ok = True
    
    for module, description in tests:
        test_cmd = f'{python_cmd} -c "import {module}; print(\\"✅ {description} OK\\")"'
        
        try:
            result = subprocess.run(test_cmd, shell=True, check=True, capture_output=True, text=True)
            print(f"{Colors.GREEN}{result.stdout.strip()}{Colors.END}")
        except subprocess.CalledProcessError:
            print(f"{Colors.RED}❌ Error importando {module} ({description}){Colors.END}")
            all_ok = False
    
    return all_ok

def check_model_file():
    """Verifica que el modelo esté presente"""
    print(f"{Colors.YELLOW}🧠 Verificando modelo YOLO...{Colors.END}")
    
    model_path = Path("best.blob")
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"{Colors.GREEN}✅ Modelo encontrado: best.blob ({size_mb:.1f} MB){Colors.END}")
        return True
    else:
        print(f"{Colors.RED}❌ Archivo best.blob no encontrado{Colors.END}")
        print(f"{Colors.RED}   Asegúrate de tener el modelo en la carpeta del proyecto{Colors.END}")
        return False

def show_usage_instructions():
    """Muestra instrucciones de uso"""
    print(f"\n{Colors.CYAN}{Colors.BOLD}🎉 ¡INSTALACIÓN COMPLETADA!{Colors.END}")
    print(f"{Colors.GREEN}=" * 50 + f"{Colors.END}")
    
    system = platform.system()
    if system == "Windows":
        activate_cmd = ".venv\\Scripts\\activate"
        run_cmd = "python detector_flores.py"
    else:
        activate_cmd = "source .venv/bin/activate"
        run_cmd = "python detector_flores.py"
    
    print(f"\n{Colors.BOLD}📋 INSTRUCCIONES DE USO:{Colors.END}")
    print(f"1. Conecta tu cámara OAK-1 al puerto USB")
    print(f"2. Activa el entorno virtual:")
    print(f"   {Colors.CYAN}{activate_cmd}{Colors.END}")
    print(f"3. Ejecuta el detector:")
    print(f"   {Colors.CYAN}{run_cmd}{Colors.END}")
    
    print(f"\n{Colors.BOLD}🎮 CONTROLES:{Colors.END}")
    print(f"• {Colors.CYAN}Q{Colors.END} - Salir")
    print(f"• {Colors.CYAN}R{Colors.END} - Reiniciar estadísticas")
    print(f"• {Colors.CYAN}ESPACIO{Colors.END} - Pausar/reanudar")
    print(f"• {Colors.CYAN}+/-{Colors.END} - Ajustar umbral de confianza")
    
    print(f"\n{Colors.BOLD}📚 DOCUMENTACIÓN:{Colors.END}")
    print(f"• README.md - Guía completa")
    print(f"• CHANGELOG.md - Historial de versiones")
    
    print(f"\n{Colors.PURPLE}🌐 GitHub: https://github.com/cristianquiroz6211/detector-flores-oak1{Colors.END}")
    print(f"{Colors.PURPLE}👤 Autor: Cristian Quiroz{Colors.END}")

def main():
    """Función principal del instalador"""
    print_header()
    
    # Verificaciones del sistema
    if not check_python_version():
        sys.exit(1)
    
    if not check_system():
        sys.exit(1)
    
    # Crear entorno virtual
    if not create_virtual_environment():
        sys.exit(1)
    
    # Instalar dependencias
    if not activate_and_install():
        print(f"\n{Colors.RED}❌ Error durante la instalación de dependencias{Colors.END}")
        print(f"{Colors.YELLOW}💡 Posibles soluciones:{Colors.END}")
        print("• Verifica tu conexión a internet")
        print("• Ejecuta como administrador/sudo si es necesario")
        print("• Instala Visual C++ Build Tools en Windows")
        sys.exit(1)
    
    # Verificar instalación
    if not verify_installation():
        print(f"\n{Colors.YELLOW}⚠️  Advertencia: Algunas dependencias podrían tener problemas{Colors.END}")
        print("Intenta ejecutar el programa y verifica los errores específicos")
    
    # Verificar modelo
    if not check_model_file():
        print(f"\n{Colors.YELLOW}⚠️  Advertencia: Modelo no encontrado{Colors.END}")
        print("El programa necesita el archivo best.blob para funcionar")
    
    # Mostrar instrucciones finales
    show_usage_instructions()

if __name__ == "__main__":
    main()
