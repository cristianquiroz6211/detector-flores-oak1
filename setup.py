#!/usr/bin/env python3
"""
Configuración de instalación para Detector de Flores OAK-1

Este archivo permite instalar el proyecto como un paquete Python
y define metadatos importantes del proyecto.

Uso:
    pip install -e .  # Instalación en modo desarrollo
    pip install .     # Instalación normal
"""

from setuptools import setup, find_packages

# Leer el README para la descripción larga
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Leer los requisitos desde requirements.txt
def parse_requirements():
    """Extrae dependencias principales del archivo requirements.txt"""
    requirements = []
    try:
        with open("requirements.txt", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # Ignorar comentarios y líneas vacías
                if line and not line.startswith("#"):
                    # Extraer solo el nombre del paquete sin comentarios
                    if "#" in line:
                        line = line.split("#")[0].strip()
                    if line:  # Si aún hay contenido después de remover comentarios
                        requirements.append(line)
    except FileNotFoundError:
        # Dependencias básicas si no existe requirements.txt
        requirements = [
            "depthai>=2.24.0.0",
            "opencv-python>=4.9.0.80",
            "numpy>=1.26.4",
            "Pillow>=10.3.0"
        ]
    return requirements

setup(
    # === INFORMACIÓN BÁSICA ===
    name="detector-flores-oak1",
    version="1.0.0",
    author="Cristian Quiroz",
    author_email="cristianquiroz6211@gmail.com",
    
    # === DESCRIPCIÓN ===
    description="Sistema avanzado de detección y clasificación de flores para cámara OAK-1",
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    # === URLS DEL PROYECTO ===
    url="https://github.com/cristianquiroz6211/detector-flores-oak1",
    project_urls={
        "Bug Tracker": "https://github.com/cristianquiroz6211/detector-flores-oak1/issues",
        "Documentation": "https://github.com/cristianquiroz6211/detector-flores-oak1#readme",
        "Source Code": "https://github.com/cristianquiroz6211/detector-flores-oak1",
    },
    
    # === CLASIFICACIÓN ===
    classifiers=[
        # Estado de desarrollo
        "Development Status :: 4 - Beta",
        
        # Audiencia objetivo
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: End Users/Desktop",
        
        # Temática
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Multimedia :: Graphics :: Capture :: Digital Camera",
        "Topic :: Software Development :: Libraries :: Python Modules",
        
        # Licencia
        "License :: OSI Approved :: MIT License",
        
        # Compatibilidad de Python
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        
        # Sistema operativo
        "Operating System :: OS Independent",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
    ],
    
    # === ESTRUCTURA DEL PAQUETE ===
    packages=find_packages(),
    python_requires=">=3.8",
    
    # === DEPENDENCIAS ===
    install_requires=parse_requirements(),
    
    # Dependencias opcionales
    extras_require={
        "dev": [
            "pytest>=8.2.0",
            "black>=24.4.0",
            "flake8>=7.0.0",
        ],
        "docs": [
            "sphinx>=7.2.6",
            "sphinx-rtd-theme>=2.0.0",
        ]
    },
    
    # === ARCHIVOS INCLUIDOS ===
    package_data={
        "": ["*.md", "*.txt", "*.blob"],
    },
    include_package_data=True,
    
    # === SCRIPTS EJECUTABLES ===
    entry_points={
        "console_scripts": [
            "detector-flores=detector_flores:main",
        ],
    },
    
    # === PALABRAS CLAVE ===
    keywords=[
        "computer-vision", "yolo", "oak-1", "depthai", 
        "flower-detection", "real-time", "agriculture", 
        "opencv", "artificial-intelligence", "machine-learning"
    ],
    
    # === CONFIGURACIÓN ADICIONAL ===
    zip_safe=False,
    platforms=["any"],
    
    # Requerir archivos específicos
    data_files=[
        ("", ["LICENSE", "README.md", "requirements.txt"]),
    ],
)
